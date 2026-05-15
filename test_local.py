"""
Test script to validate the SHL Assessment Chatbot locally.
Run with: python test_local.py
"""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.models import Message


async def test_conversation(agent, messages_data, test_name="Test"):
    """Run a test conversation."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    messages = [Message(**m) for m in messages_data]

    for i, msg in enumerate(messages):
        if msg.role == "user":
            print(f"\nUser: {msg.content}")

    print(f"\n--- Sending {len(messages)} messages ---")
    response = await agent.chat(messages)

    print(f"\nAgent reply: {response.reply}")
    if response.recommendations:
        print(f"\nRecommendations ({len(response.recommendations)}):")
        for j, rec in enumerate(response.recommendations, 1):
            print(f"  {j}. {rec.name} ({rec.test_type}) - {rec.url}")
    else:
        print("\nNo recommendations (still gathering context or refusing).")
    print(f"\nEnd of conversation: {response.end_of_conversation}")

    # Validate response schema
    response_dict = response.model_dump()
    assert "reply" in response_dict, "Missing 'reply' field"
    assert "recommendations" in response_dict, "Missing 'recommendations' field"
    assert "end_of_conversation" in response_dict, "Missing 'end_of_conversation' field"
    assert isinstance(response_dict["recommendations"], list), "recommendations must be a list"

    for rec in response_dict["recommendations"]:
        assert "name" in rec, "Recommendation missing 'name'"
        assert "url" in rec, "Recommendation missing 'url'"
        assert "test_type" in rec, "Recommendation missing 'test_type'"
        assert rec["url"].startswith("https://www.shl.com/"), f"Invalid URL: {rec['url']}"

    print("[PASS] Schema validation passed!")
    return response


async def main():
    from app.agent import SHLAgent

    print("Initializing agent...")
    agent = SHLAgent()
    print("Agent initialized!")

    # Test 1: Vague query — should clarify
    await test_conversation(agent, [
        {"role": "user", "content": "I need an assessment"}
    ], "Vague Query - Should Clarify")

    # Test 2: Specific query — should recommend
    await test_conversation(agent, [
        {"role": "user", "content": "I'm hiring a senior Java developer who works with stakeholders"},
        {"role": "assistant", "content": "What is the seniority level?"},
        {"role": "user", "content": "Mid-level, around 4 years experience"}
    ], "Java Developer - Should Recommend")

    # Test 3: Contact center screening
    await test_conversation(agent, [
        {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus in English US."}
    ], "Contact Center - Detailed Query")

    # Test 4: Off-topic — should refuse
    await test_conversation(agent, [
        {"role": "user", "content": "What salary should I offer a Java developer?"}
    ], "Off-topic - Should Refuse")

    # Test 5: Comparison
    await test_conversation(agent, [
        {"role": "user", "content": "What's the difference between OPQ32r and Verify G+?"}
    ], "Comparison - Should Compare")

    # Test 6: Refinement
    await test_conversation(agent, [
        {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word"},
        {"role": "assistant", "content": "For a quick screening of admin assistants on Excel and Word skills, here are my recommendations."},
        {"role": "user", "content": "Actually, add a simulation too - we want to capture capabilities"}
    ], "Refinement - Should Update Shortlist")

    # Test 7: Senior leadership
    await test_conversation(agent, [
        {"role": "user", "content": "We need a solution for senior leadership."},
        {"role": "assistant", "content": "Happy to help. Who is this meant for?"},
        {"role": "user", "content": "CXOs, director-level positions; people with more than 15 years of experience. Selection — comparing candidates against a leadership benchmark."}
    ], "Senior Leadership Selection")

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
