"""Airport reference geometry used by the OSM and OpenSky collectors."""
from __future__ import annotations

# (south, west, north, east) bounding boxes enclosing each aerodrome
AIRPORTS = {
    "JFK": {"name": "John F. Kennedy International", "icao": "KJFK",
            "bbox": (40.615, -73.83, 40.665, -73.74), "center": (40.6413, -73.7781)},
    "ATL": {"name": "Hartsfield-Jackson Atlanta International", "icao": "KATL",
            "bbox": (33.62, -84.455, 33.66, -84.40), "center": (33.6407, -84.4277)},
    "DCA": {"name": "Ronald Reagan Washington National", "icao": "KDCA",
            "bbox": (38.84, -77.05, 38.862, -77.03), "center": (38.8512, -77.0402)},
}
