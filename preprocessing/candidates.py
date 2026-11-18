
from pathlib import Path
import csv
import math
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

# Dataset location
DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_RADIUS_M = 100
MAX_CANDIDATES = 8

PARTITIONS = ["train", "student_test"]

COLUMNS = [
    "partition",
    "trip_id",
    "observation_id",
    "arc_id",
    "longitude",
    "latitude",
    "timestamp",
    "distance_m",
    "gps_heading",
    "road_heading",
    "heading_difference",
    "gps_speed_mps",
    "projected_x",
    "projected_y",
    "crs",
    "fallback_candidate"
]


def read_file(path, columns):
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)

    if data.shape[1] != columns:
        raise ValueError(
            f"Unexpected columns in {path}: {data.shape[1]}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values in {path}")

    return data


def get_transformer(longitude, latitude):
    zone = int((longitude + 180) // 6) + 1
    zone = max(1, min(zone, 60))

    epsg = (
        32600 if latitude >= 0 else 32700
    ) + zone

    crs = CRS.from_epsg(epsg)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    return transformer, crs.to_string()


def bearing(x1, y1, x2, y2):
    angle = math.degrees(
        math.atan2(x2 - x1, y2 - y1)
    )
    return (angle + 360) % 360


def angle_difference(a, b):
    difference = abs(a - b)
    return min(difference, 360 - difference)


def get_candidates(tree, lines, original_ids,
                   id_lookup, point, start_x, start_y,
                   end_x, end_y, gps_heading):

    matches = tree.query(
        point.buffer(SEARCH_RADIUS_M)
    )

    candidates = []

    for match in matches:

        if isinstance(match, (int, np.integer)):
            local_index = int(match)
            line = lines[local_index]
            arc_id = original_ids[local_index]

        else:
            line = match
            local_index = id_lookup[id(line)]
            arc_id = original_ids[local_index]

        distance = point.distance(line)

        if distance <= SEARCH_RADIUS_M:

            road_heading = bearing(
                start_x[arc_id],
                start_y[arc_id],
                end_x[arc_id],
                end_y[arc_id]
            )

            difference = (
                angle_difference(gps_heading, road_heading)
                if np.isfinite(gps_heading)
                else np.nan
            )

            candidates.append((
                distance,
                arc_id,
                road_heading,
                difference
            ))

    candidates.sort(key=lambda item: item[0])

    return candidates[:MAX_CANDIDATES]


def process_trip(trip_folder, partition, writer):

    trip_id = trip_folder.name

    track = read_file(
        trip_folder / f"{trip_id}.track", 3
    )

    nodes = read_file(
        trip_folder / f"{trip_id}.nodes", 2
    )

    arcs_raw = read_file(
        trip_folder / f"{trip_id}.arcs", 2
    )

    # Validate arc endpoint IDs
    if not np.all(np.isclose(arcs_raw, np.round(arcs_raw))):
        raise ValueError("Arc endpoints must be integers")

    arcs = np.round(arcs_raw).astype(np.int64)

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("Invalid node IDs in arcs")

    # Project coordinates into UTM metres
    center_lon = float(np.median(nodes[:, 0]))
    center_lat = float(np.median(nodes[:, 1]))

    transformer, crs_name = get_transformer(
        center_lon, center_lat
    )

    node_x, node_y = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0], track[:, 1]
    )

    gps_x = np.asarray(gps_x)
    gps_y = np.asarray(gps_y)

    # GPS heading and speed
    count = len(track)

    headings = np.full(count, np.nan)
    speeds = np.full(count, np.nan)

    for i in range(count - 1):

        dx = gps_x[i + 1] - gps_x[i]
        dy = gps_y[i + 1] - gps_y[i]

        dt = track[i + 1, 2] - track[i, 2]

        headings[i] = bearing(
            gps_x[i], gps_y[i],
            gps_x[i + 1], gps_y[i + 1]
        )

        if dt > 0:
            speeds[i] = math.hypot(dx, dy) / dt

    # Build road segments
    start_x = node_x[arcs[:, 0]]
    start_y = node_y[arcs[:, 0]]

    end_x = node_x[arcs[:, 1]]
    end_y = node_y[arcs[:, 1]]

    lines = []
    original_ids = []

    for arc_id in range(len(arcs)):

        line = LineString([
            (start_x[arc_id], start_y[arc_id]),
            (end_x[arc_id], end_y[arc_id])
        ])

        if line.is_valid and line.length > 0:
            lines.append(line)
            original_ids.append(arc_id)

    if not lines:
        raise ValueError("No valid road segments")

    tree = STRtree(lines)

    id_lookup = {
        id(line): index
        for index, line in enumerate(lines)
    }

    # Generate candidates for every GPS observation
    for obs_id in range(count):

        point = Point(gps_x[obs_id], gps_y[obs_id])

        candidates = get_candidates(
            tree,
            lines,
            original_ids,
            id_lookup,
            point,
            start_x,
            start_y,
            end_x,
            end_y,
            headings[obs_id]
        )

        # Fallback if no candidate is within 100 metres
        fallback = False

        if not candidates:

            nearest = tree.nearest(point)

            if isinstance(nearest, (int, np.integer)):
                local_index = int(nearest)
                line = lines[local_index]
                arc_id = original_ids[local_index]
            else:
                line = nearest
                local_index = id_lookup[id(line)]
                arc_id = original_ids[local_index]

            distance = point.distance(line)

            road_heading = bearing(
                start_x[arc_id],
                start_y[arc_id],
                end_x[arc_id],
                end_y[arc_id]
            )

            difference = (
                angle_difference(
                    headings[obs_id], road_heading
                )
                if np.isfinite(headings[obs_id])
                else np.nan
            )

            candidates = [(
                distance,
                arc_id,
                road_heading,
                difference
            )]

            fallback = True

        # Write candidate features
        for distance, arc_id, road_heading, difference in candidates:

            writer.writerow({
                "partition": partition,
                "trip_id": trip_id,
                "observation_id": obs_id,
                "arc_id": arc_id,
                "longitude": track[obs_id, 0],
                "latitude": track[obs_id, 1],
                "timestamp": track[obs_id, 2],
                "distance_m": round(float(distance), 3),
                "gps_heading": (
                    round(float(headings[obs_id]), 3)
                    if np.isfinite(headings[obs_id])
                    else ""
                ),
                "road_heading": round(float(road_heading), 3),
                "heading_difference": (
                    round(float(difference), 3)
                    if np.isfinite(difference)
                    else ""
                ),
                "gps_speed_mps": (
                    round(float(speeds[obs_id]), 3)
                    if np.isfinite(speeds[obs_id])
                    else ""
                ),
                "projected_x": round(float(gps_x[obs_id]), 3),
                "projected_y": round(float(gps_y[obs_id]), 3),
                "crs": crs_name,
                "fallback_candidate": fallback
            })

    return count


def main():

    for partition in PARTITIONS:

        partition_path = DATASET_ROOT / partition

        if not partition_path.exists():
            print("Missing partition:", partition)
            continue

        trip_folders = sorted(
            p for p in partition_path.iterdir()
            if p.is_dir()
        )

        output_file = (
            OUTPUT_DIR / f"candidates_{partition}.csv"
        )

        successful = 0
        failed = 0
        total_observations = 0

        print(f"\nProcessing {partition}...")

        with output_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=COLUMNS
            )

            writer.writeheader()

            for index, trip_folder in enumerate(
                trip_folders, start=1
            ):

                try:
                    count = process_trip(
                        trip_folder,
                        partition,
                        writer
                    )

                    successful += 1
                    total_observations += count

                except Exception as error:
                    failed += 1
                    print(
                        f"Failed {trip_folder.name}: {error}"
                    )

                if index % 10 == 0:
                    print(
                        f"Processed {index}/{len(trip_folders)}"
                    )

        print(f"\n{partition.upper()} SUMMARY")
        print("Successful trips:", successful)
        print("Failed trips:", failed)
        print("GPS observations:", total_observations)
        print("Output:", output_file)


if __name__ == "__main__":
    main()