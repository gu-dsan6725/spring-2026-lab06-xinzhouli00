"""
Week 4 Lab: World Bank Data MCP Server

An MCP server that exposes:
- Resources: Local World Bank indicator data from CSV
- Tools: Live data from REST Countries and World Bank APIs

Transport: Streamable HTTP on port 8765
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import httpx
import polars as pl
from mcp.server.fastmcp import FastMCP


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE: Path = Path(__file__).parent / "data" / "world_bank_indicators.csv"
HOST: str = "127.0.0.1"
PORT: int = 8765

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    "world-bank-server",
    host=HOST,
    port=PORT,
)


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================

def _load_data() -> pl.DataFrame:
    """Load the World Bank indicators CSV file."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    return pl.read_csv(DATA_FILE)


def _fetch_rest_countries(country_code: str) -> dict:
    """Fetch country info from REST Countries API."""
    url = f"https://restcountries.com/v3.1/alpha/{country_code}"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()[0]


def _fetch_world_bank_indicator(
    country_code: str,
    indicator: str,
    year: Optional[int] = None,
) -> list:
    """Fetch indicator from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
    params = {"format": "json", "per_page": 100}
    if year:
        params["date"] = str(year)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2 or not data[1]:
            return []
        return data[1]


# =============================================================================
# PART 1: RESOURCES (Local Data)
# =============================================================================

@mcp.resource("data://schema")
def get_schema() -> str:
    """
    Return the schema of the World Bank dataset.

    This resource is provided as an example - it's already implemented.
    """
    df = _load_data()
    schema_info = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return json.dumps(schema_info, indent=2)


@mcp.resource("data://countries")
def get_countries() -> str:
    """
    List all unique countries in the dataset.

    TODO: Implement this resource.

    Hints:
    - Use _load_data() to get the DataFrame
    - Use df.select() to pick relevant columns (countryiso3code, country)
    - Use df.unique() to get unique values
    - Return as JSON string using json.dumps() or df.write_json()

    Expected output format:
    [
        {"countryiso3code": "USA", "country": "United States"},
        {"countryiso3code": "CHN", "country": "China"},
        ...
    ]
    """
    df = _load_data()
    return (
        df.select(["countryiso3code", "country"])
        .unique(maintain_order=False)
        .write_json()
    )


@mcp.resource("data://indicators/{country_code}")
def get_country_indicators(country_code: str) -> str:
    """
    Get all indicators for a specific country from local data.

    TODO: Implement this resource.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g., "USA", "CHN", "DEU")

    Hints:
    - Use _load_data() to get the DataFrame
    - Use df.filter(pl.col("countryiso3code") == country_code) to filter
    - Return as JSON string
    - Handle case where country_code is not found (return error message)

    Expected output: JSON array of indicator records for that country
    """
    df = _load_data()
    df_filtered = df.filter(pl.col("countryiso3code") == country_code)
    if df_filtered.is_empty():
        raise ValueError(f"Country code '{country_code}' not found in dataset")
    return df_filtered.write_json()

# =============================================================================
# PART 2: TOOLS (External APIs)
# =============================================================================

@mcp.tool()
def get_country_info(country_code: str) -> dict:
    """
    Fetch detailed information about a country from REST Countries API.

    TODO: Implement this tool.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code (e.g., "US", "USA", "DE")

    Returns:
        Dictionary with country information including:
        - name: Common name of the country
        - capital: Capital city
        - region: Geographic region (e.g., "Americas", "Europe")
        - subregion: Geographic subregion
        - languages: List of official languages
        - currencies: List of currency codes
        - population: Current population
        - flag: Flag emoji

    Hints:
    - Use _fetch_rest_countries(country_code) to get raw API data
    - Extract the relevant fields from the response
    - Handle errors gracefully (invalid country code, API failure)

    API Response structure (key fields):
    - name.common: "United States"
    - capital: ["Washington, D.C."]
    - region: "Americas"
    - subregion: "North America"
    - languages: {"eng": "English"}
    - currencies: {"USD": {"name": "United States dollar", ...}}
    - population: 331002651
    - flag: "🇺🇸"
    """
    logger.info(f"Fetching country info for: {country_code}")
    # TODO: Implement using _fetch_rest_countries()
    try:
        data = _fetch_rest_countries(country_code)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        logger.exception("Failed to fetch country info for %s", country_code)
        return {"country_code": country_code, "error": str(e)}

    languages = data.get("languages") or {}
    currencies = data.get("currencies") or {}
    capital = data.get("capital") or []
 
    return {
         "name": data.get("name", {}).get("common"),
         "capital": capital[0] if capital else None,
         "region": data.get("region"),
         "subregion": data.get("subregion"),
         "languages": list(languages.values()),
         "currencies": list(currencies.keys()),
         "population": data.get("population"),
         "flag": data.get("flag"),
     }
 



@mcp.tool()
def get_live_indicator(
    country_code: str,
    indicator: str,
    year: int = 2022,
) -> dict:
    """
    Fetch a specific indicator value from the World Bank API.

    TODO: Implement this tool.

    Args:
        country_code: ISO 3166-1 alpha-2 or alpha-3 country code
        indicator: World Bank indicator ID (e.g., "NY.GDP.PCAP.CD" for GDP per capita)
        year: Year to fetch data for (default: 2022)

    Returns:
        Dictionary with:
        - country: Country code
        - country_name: Full country name
        - indicator: Indicator ID
        - indicator_name: Human-readable indicator name
        - year: Year of data
        - value: The indicator value

    Common indicators:
        - NY.GDP.PCAP.CD: GDP per capita (current US$)
        - SP.POP.TOTL: Total population
        - SP.DYN.LE00.IN: Life expectancy at birth
        - SE.ADT.LITR.ZS: Adult literacy rate

    Hints:
    - Use _fetch_world_bank_indicator(country_code, indicator, year)
    - The API returns a list; find the entry matching the requested year
    - Handle case where no data exists for that year
    """
    logger.info(f"Fetching {indicator} for {country_code} in {year}")
    # TODO: Implement using _fetch_world_bank_indicator()
    try:
        data = _fetch_world_bank_indicator(country_code, indicator, year)
    except (httpx.HTTPError) as e:
        logger.exception("Failed to fetch World Bank indicator data for %s", country_code)
        return {
            "country": country_code,
            "indicator": indicator,
            "year": year,
            "value": None,
            "error": str(e),
        }
    if not data:
        return {
            "country": country_code,
            "indicator": indicator,
            "year": year,
            "value": None,
            "error": f"No data found for {indicator} in {country_code} for year {year}",
        }
    entry = next((item for item in data if item.get("date") == str(year)), None)
    record = entry if entry else data[0]
    return {
        "country": record.get("country", {}).get("id"),
        "country_name": record.get("country", {}).get("value"),
        "indicator": record.get("indicator", {}).get("id"),
            "indicator_name": record.get("indicator", {}).get("value"),
            "year": record.get("date"),
            "value": record.get("value"),
        }
    


@mcp.tool()
def compare_countries(
    country_codes: list[str],
    indicator: str,
    year: int = 2022,
) -> list[dict]:
    """
    Compare an indicator across multiple countries.

    TODO: Implement this tool.

    Args:
        country_codes: List of ISO country codes to compare (e.g., ["USA", "CHN", "DEU"])
        indicator: World Bank indicator ID to compare
        year: Year to fetch data for

    Returns:
        List of dictionaries, one per country, each containing:
        - country: Country code
        - country_name: Full country name
        - indicator: Indicator ID
        - year: Year
        - value: The indicator value (or None if not available)

    Hints:
    - Loop through country_codes and call get_live_indicator() for each
    - Collect results into a list
    - Handle errors for individual countries (don't fail the whole request)
    """
    logger.info(f"Comparing {indicator} for countries: {country_codes}")
    if not country_codes:
        return []

    results_by_index: dict[int, dict] = {}
    max_workers = min(32, len(country_codes))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(get_live_indicator, code, indicator, year): (idx, code)
            for idx, code in enumerate(country_codes)
        }
        for future in as_completed(future_to_item):
            idx, code = future_to_item[future]
            try:
                results_by_index[idx] = future.result()
            except Exception as e:
                logger.exception("Failed comparing indicator for %s", code)
                results_by_index[idx] = {
                    "country": code,
                    "indicator": indicator,
                    "year": year,
                    "value": None,
                    "error": str(e),
                }

    return [results_by_index[i] for i in range(len(country_codes))]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting World Bank MCP Server on http://{HOST}:{PORT}/mcp")
    logger.info(f"Connect with MCP Inspector or test client at http://{HOST}:{PORT}/mcp")
    logger.info("Press Ctrl+C to stop")
    mcp.run(transport="streamable-http")
