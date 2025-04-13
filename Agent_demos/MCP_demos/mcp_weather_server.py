####
# 1. Open bash terminal and cd ~/Documents/github_christy_scratch/temporal_demos
# 2. Run this code (MCP server) with command: uv run --active mcp_weather_server.py
# 3. Edit /Users/christy/Library/Application Support/Claude/claude_desktop_config.json
# 4. Open another bash terminal and tail -n 20 -F ~/Library/Logs/Claude/mcp*.log
# 5. Run MCP client from Claude Desktop.  Make sure you see tools and server first.
####
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

# Define Python helper functions
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

# Add Tools
# Each Tool requires: 
# - function_name with return type, 
# - params with types, 
# - docstring
# - fully encapsulated error handling
@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location. Provides a detailed 7-day forecast.

    For a full 7-day forecast, please specify '7-day forecast' in your request.
    For a shorter forecast, just ask for example: 'weather san francisco'.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the NWS grid endpoint url
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)
    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast from the grid url
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)
    if not forecast_data:
        return "Unable to fetch detailed forecast."
    
    # Each "day" consists of 2 periods: 6am-6pm, 6pm-6am
    periods = forecast_data["properties"]["periods"][:14]  # 7 day forecast
    forecasts = []
    
    # Format into a readable daily forecast
    # Process periods in pairs (day and night)
    for i in range(0, len(periods), 2):
        day_period = periods[i]
        night_period = periods[i + 1] if i + 1 < len(periods) else None
        
        # Get the day name and date
        start_time = datetime.fromisoformat(day_period['startTime'].replace('Z', '+00:00'))
        day_name = day_period['name']
        # if day_name.lower() == 'today':
        #     day_name = start_time.strftime('%A')
        formatted_date = start_time.strftime('%m-%d-%Y')
        
        # Calculate temperature range
        max_temp = day_period['temperature']
        min_temp = night_period['temperature'] if night_period else max_temp
        temp_unit = day_period['temperatureUnit']
            
        # Combine forecasts, prioritizing day period but including night details
        detailed_forecast = day_period['detailedForecast']
        if night_period:
            detailed_forecast += f"\nNight: {night_period['detailedForecast']}"
        
        forecast = f"""
{day_name}: {formatted_date}
Temperature: {min_temp}°{temp_unit} to {max_temp}°{temp_unit}
Forecast: {detailed_forecast}
"""
        forecasts.append(forecast)
    return "\n---\n".join(forecasts)

if __name__ == "__main__":

    # Initialize and run the server. Registers tools decorated with @mcp.tool().
    mcp.run(transport='stdio')