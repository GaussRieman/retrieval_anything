from typing import List, Dict
import re
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Math")

# === MCP TOOL 1: Markdown Structure Extractor ===
@mcp.tool()
def extract_structure_from_markdown(markdown: str) -> Dict:
    """Extracts titles, sections, bullet points, and emphasis from a markdown doc"""
    headers = re.findall(r"^#{1,6} .*", markdown, re.MULTILINE)
    bullets = re.findall(r"^- .+", markdown, re.MULTILINE)
    bolds = re.findall(r"\*\*(.*?)\*\*", markdown)
    italics = re.findall(r"\*(.*?)\*", markdown)
    return {
        "headers": headers,
        "bullets": bullets,
        "bold_text": bolds,
        "italic_text": italics,
    }

    
# === MCP TOOL 2: Insight Analyzer ===
@mcp.tool()
def analyze_content(structure: Dict) -> List[str]:
    """Analyzes the extracted structure and generates insights"""
    insights = []
    if structure["headers"]:
        insights.append(f"Document is structured into {len(structure['headers'])} sections.")
    if len(structure["bold_text"]) > 3:
        insights.append("The document uses a strong emphasis on certain points.")
    if any("conclusion" in h.lower() for h in structure["headers"]):
        insights.append("There is a section dedicated to conclusions.")
    if any("future work" in h.lower() for h in structure["headers"]):
        insights.append("Author discusses future directions.")
    if len(structure["bullets"]) > 5:
        insights.append("Many bullet points suggest detailed breakdowns or lists.")
    return insights


# === MCP TOOL 3: Summary Generator ===
@mcp.tool()
def generate_summary(insights: List[str]) -> str:
    """Generates a natural language summary from the insights"""
    if not insights:
        return "No significant insights could be derived."
    summary = "Here’s a quick summary of the document:\n"
    for i, line in enumerate(insights, 1):
        summary += f"{i}. {line}\n"
    return summary

# === PROMPT to Trigger Agent ===
# User message (you can simulate this):
# "Take a look at this markdown report and give me a short summary of its structure and main takeaways."
# The agent will:
# - Call extract_structure_from_markdown()
# - Then analyze_content()
# - Then generate_summary()
# And return the final summary to the user.

# Now you just need to feed markdown text, and let the agent flow run 🤖


if __name__ == "__main__":
    print("Starting Markdown MCP server...")
    mcp.run(transport="stdio")