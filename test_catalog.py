"""Quick test of catalog search."""
from app.catalog import AssessmentCatalog

c = AssessmentCatalog()

# Test 1: Java developer
print("=== Java Developer ===")
results = c.search("Java developer senior", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']}) score={r['relevance_score']:.3f}")

# Test 2: Contact center
print("\n=== Contact Center ===")
results = c.search("contact center agent customer service entry level", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']}) score={r['relevance_score']:.3f}")

# Test 3: Leadership
print("\n=== Senior Leadership ===")
results = c.search("senior leadership CXO director executive personality", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']}) score={r['relevance_score']:.3f}")

# Test 4: Safety
print("\n=== Safety/Plant Operators ===")
results = c.search("plant operators safety dependability chemical facility", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']}) score={r['relevance_score']:.3f}")

# Test 5: Excel Word admin
print("\n=== Admin Excel Word ===")
results = c.search("admin assistant Excel Word office skills", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']}) score={r['relevance_score']:.3f}")

# Test name lookup
print("\n=== Name Lookup ===")
opq = c.get_by_name("Occupational Personality Questionnaire OPQ32r")
if opq:
    print(f"  Found: {opq['name']} -> {opq['link']}")
else:
    print("  OPQ32r not found!")
