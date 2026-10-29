# Data

`cneos_fireballs_2026-10-29.csv` is the complete NASA JPL CNEOS fireball table (1,074 events, 1988-04-15 to 2026-10-04), downloaded on 2026-10-29 from the Fireball Data API:

    https://ssd-api.jpl.nasa.gov/fireball.api?vel-comp=true

It's saved exactly as the API returned it (API version 1.2): every value is the API's string, and an empty cell is a null. Field definitions are in the [API documentation](https://ssd-api.jpl.nasa.gov/doc/fireball.html) and on the [CNEOS fireball page](https://cneos.jpl.nasa.gov/fireballs/intro.html). The events come from U.S. Government sensors; the data are NASA's, not mine.

`examples/fetch_data.py` downloads a fresh copy into a new dated file. Raw JSON downloads go in `data/raw/`, which isn't committed.
