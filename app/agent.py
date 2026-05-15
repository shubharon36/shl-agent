"""
Core agent logic: takes conversation history, retrieves relevant catalog items,
calls LLM (Gemini primary, Groq fallback), and returns structured response.
"""

import json
import os
import re
import time
import traceback
from typing import List, Dict, Any, Optional

from .catalog import AssessmentCatalog
from .prompts import build_system_prompt
from .models import Message, ChatResponse, Recommendation


class SHLAgent:
    """Conversational agent for SHL assessment recommendations."""

    def __init__(self):
        # Configure LLM clients
        self.gemini_client = None
        self.groq_client = None

        # Try Gemini
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                print("Gemini client initialized.")
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")

        # Try Groq as fallback
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                print("Groq client initialized.")
            except Exception as e:
                print(f"Failed to initialize Groq: {e}")

        if not self.gemini_client and not self.groq_client:
            raise ValueError("At least one of GEMINI_API_KEY or GROQ_API_KEY must be set")

        # Load catalog and build search index
        self.catalog = AssessmentCatalog()

        print("SHL Agent initialized successfully.")

    def _extract_search_queries(self, messages: List[Message]) -> List[str]:
        """Extract meaningful search queries from the conversation history."""
        queries = []
        user_messages = [m.content for m in messages if m.role == "user"]

        if user_messages:
            full_context = " ".join(user_messages)
            queries.append(full_context)
            if len(user_messages) > 1:
                queries.append(user_messages[-1])

        return queries

    def _retrieve_relevant_assessments(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Retrieve relevant assessments based on conversation context."""
        queries = self._extract_search_queries(messages)

        seen_ids = set()
        all_results = []

        for query in queries:
            results = self.catalog.search(query, top_k=25)
            for r in results:
                eid = r["entity_id"]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    all_results.append(r)

        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_results[:30]

    def _check_for_named_assessments(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Check if user mentions specific assessment names and retrieve them."""
        extra = []
        known_patterns = [
            "OPQ32r", "OPQ", "Verify G+", "Verify Interactive", "DSI",
            "GSA", "Global Skills", "Graduate Scenarios", "SVAR",
            "Automata", "Sales Transformation", "Leadership Report",
            "HIPAA", "Contact Center", "Customer Service",
        ]
        full_text = " ".join(m.content for m in messages)
        for pattern in known_patterns:
            if pattern.lower() in full_text.lower():
                results = self.catalog.search(pattern, top_k=5)
                extra.extend(results)

        return extra

    def _validate_recommendations(self, recommendations: List[Dict]) -> List[Recommendation]:
        """Validate that all recommendations exist in the catalog and have correct URLs."""
        validated = []
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            name = rec.get("name", "")
            url = rec.get("url", "")

            # Try exact name
            catalog_item = self.catalog.get_by_name(name)
            if catalog_item:
                validated.append(Recommendation(
                    name=catalog_item["name"],
                    url=catalog_item["link"],
                    test_type=catalog_item["test_type"]
                ))
                continue

            # Try by URL
            if url:
                for a in self.catalog.assessments:
                    if a["link"] == url:
                        validated.append(Recommendation(
                            name=a["name"],
                            url=a["link"],
                            test_type=a["test_type"]
                        ))
                        break
                else:
                    # Try fuzzy
                    matches = self.catalog.get_by_names([name])
                    if matches:
                        m = matches[0]
                        validated.append(Recommendation(
                            name=m["name"], url=m["link"], test_type=m["test_type"]
                        ))
            else:
                matches = self.catalog.get_by_names([name])
                if matches:
                    m = matches[0]
                    validated.append(Recommendation(
                        name=m["name"], url=m["link"], test_type=m["test_type"]
                    ))

        # Deduplicate by URL
        seen_urls = set()
        deduped = []
        for r in validated:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                deduped.append(r)

        return deduped[:10]

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, handling common formatting issues."""
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return {
            "reply": text if text else "I can help you find the right SHL assessments. Could you tell me more about the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False
        }

    def _call_gemini(self, system_prompt: str, messages: List[Message]) -> Optional[str]:
        """Call Gemini API."""
        if not self.gemini_client:
            return None

        from google.genai import types

        contents = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg.content)]
            ))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            ],
        )

        models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
        for model_name in models_to_try:
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                if response.text:
                    print(f"Gemini ({model_name}) responded successfully.")
                    return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "exhausted" in error_str.lower():
                    print(f"Gemini {model_name} quota exhausted, trying next...")
                    continue
                elif "404" in error_str:
                    print(f"Gemini {model_name} not found, trying next...")
                    continue
                else:
                    print(f"Gemini {model_name} error: {e}")
                    continue

        return None

    def _call_groq(self, system_prompt: str, messages: List[Message]) -> Optional[str]:
        """Call Groq API as fallback."""
        if not self.groq_client:
            return None

        groq_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            groq_messages.append({"role": msg.role, "content": msg.content})

        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for model_name in models_to_try:
            try:
                response = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=groq_messages,
                    temperature=0.2,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                if text:
                    print(f"Groq ({model_name}) responded successfully.")
                    return text
            except Exception as e:
                print(f"Groq {model_name} error: {e}")
                continue

        return None

    async def chat(self, messages: List[Message]) -> ChatResponse:
        """Process a conversation and return the agent's response."""
        try:
            # 1. Retrieve relevant assessments
            retrieved = self._retrieve_relevant_assessments(messages)

            # 2. Check for named assessments
            named = self._check_for_named_assessments(messages)
            seen = {a["entity_id"] for a in retrieved}
            for a in named:
                if a["entity_id"] not in seen:
                    retrieved.append(a)
                    seen.add(a["entity_id"])

            # 3. Format catalog context
            catalog_context = self.catalog.format_for_context(retrieved)

            # 4. Build system prompt
            system_prompt = build_system_prompt(catalog_context)

            # 5. Call LLM (Gemini first, then Groq fallback)
            raw_text = self._call_gemini(system_prompt, messages)
            if not raw_text:
                raw_text = self._call_groq(system_prompt, messages)
            if not raw_text:
                return ChatResponse(
                    reply="I apologize, all LLM providers are currently unavailable. Please try again in a moment.",
                    recommendations=[],
                    end_of_conversation=False
                )

            # 6. Parse response
            parsed = self._parse_llm_response(raw_text)

            # 7. Validate recommendations
            raw_recs = parsed.get("recommendations", []) or []
            validated_recs = self._validate_recommendations(raw_recs)

            # 8. Build response
            return ChatResponse(
                reply=parsed.get("reply", "I apologize, I encountered an issue. Could you rephrase your question?"),
                recommendations=validated_recs,
                end_of_conversation=bool(parsed.get("end_of_conversation", False))
            )

        except Exception as e:
            traceback.print_exc()
            return ChatResponse(
                reply="I apologize, I encountered a technical issue. Could you please try again?",
                recommendations=[],
                end_of_conversation=False
            )
