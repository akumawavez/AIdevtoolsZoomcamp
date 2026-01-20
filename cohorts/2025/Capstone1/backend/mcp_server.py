from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("AptitudeTestEvaluator")

@mcp.tool()
def grade_submission(code: str, language: str, test_cases: list[dict]) -> dict:
    """
    Grades a code submission against provided test cases.
    This is a basic implementation that would likely communicate with a sandbox in production.
    """
    results = []
    passed_count = 0
    
    # Mock grading logic
    for case in test_cases:
        # In a real scenario, this would execute the code
        is_correct = True # Mock result
        if is_correct:
            passed_count += 1
        results.append({"input": case.get("input"), "passed": is_correct})
        
    score = (passed_count / len(test_cases)) * 100 if test_cases else 0
    
    return {
        "score": score,
        "details": results,
        "feedback": "Great job! (Mock feedback)"
    }

if __name__ == "__main__":
    mcp.run()
