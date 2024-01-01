import re


def extract_spec_from_string(string: str) -> str:
    result = "{}"
    pattern = r"st\.vega_lite_chart\(\s*df,\s*({.*?}),\s*(?:use_container_width=(True|False),\s*)?\)"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        result = match.group(1)
    return result
