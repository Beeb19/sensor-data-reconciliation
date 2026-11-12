FROM python:3.11-slim

WORKDIR /app

# Standard data-science tooling for the agent (a pure-Python solution also works).
RUN pip install pandas==3.0.3 numpy==2.4.6

# Seed inputs the agent reads at runtime. The build context is the environment/
# directory, so the sources are data/... (not environment/data/...). They land at
# the /app/ top level to match the paths used in instruction.md.
COPY data/sensor_alpha.log /app/
COPY data/sensor_beta.csv /app/
