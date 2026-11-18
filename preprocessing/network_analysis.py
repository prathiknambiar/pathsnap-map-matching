
from pathlib import Path
import csv
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DATASET_ROOT = Path(
    r"C:\Users\megha\Downloads\Student folder\Student folder"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITIONS = ["train", "student_test"]


def load_data(path, columns):
    data = np.loadtxt(path, dtype=float)
    data = np.atleast_2d(data)

    if data.shape[1] != columns:
        raise ValueError(
            f"Expected {columns} columns in {path}, "
            f"found {data.shape[1]}"
        )

    return data


def load_route(path):
    # Route files may contain multiple IDs per line.
    # Flatten all whitespace-separated values into one sequence.
    return np.loadtxt(path, dtype=float).reshape(-1)


def analyse_trip(folder, partition):
    trip_id = folder.name

    nodes = load_data(
        folder / f"{trip_id}.nodes", 2
    )

    arcs_raw = load_data(
        folder / f"{trip_id}.arcs", 2
    )

    route_raw = load_route(
        folder / f"{trip_id}.route"
    )

    if not np.all(np.isclose(arcs_raw, np.round(arcs_raw))):
        raise ValueError("Non-integer arc endpoints")

    arcs = np.round(arcs_raw).astype(np.int64)

    result = {
        "partition": partition,
        "trip_id": trip_id,
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_segments": len(route_raw)
    }

    # Validate arc endpoints
    valid = (
        (arcs[:, 0] >= 0) &
        (arcs[:, 0] < len(nodes)) &
        (arcs[:, 1] >= 0) &
        (arcs[:, 1] < len(nodes))
    )

    result["invalid_arcs"] = int((~valid).sum())

    valid_arcs = arcs[valid]

    # Calculate incoming and outgoing degrees
    incoming = np.bincount(
        valid_arcs[:, 1],
        minlength=len(nodes)
    )

    outgoing = np.bincount(
        valid_arcs[:, 0],
        minlength=len(nodes)
    )

    degree = incoming + outgoing

    result["isolated_nodes"] = int(
        np.sum(degree == 0)
    )

    result["connected_nodes"] = int(
        np.sum(degree > 0)
    )

    # Create sparse adjacency matrix
    adjacency = csr_matrix(
        (
            np.ones(len(valid_arcs), dtype=np.int8),
            (valid_arcs[:, 0], valid_arcs[:, 1])
        ),
        shape=(len(nodes), len(nodes))
    )

    # Calculate weakly connected components
    component_count, labels = connected_components(
        adjacency,
        directed=True,
        connection="weak",
        return_labels=True
    )

    component_sizes = np.bincount(labels)

    result["connected_components"] = int(component_count)

    result["largest_component_percentage"] = round(
        100 * component_sizes.max() / len(nodes), 2
    )

    # Validate reference route arc IDs
    route_integer = np.all(
        np.isclose(route_raw, np.round(route_raw))
    )

    if not route_integer:
        result["invalid_route_ids"] = "Non-integer"
        result["route_discontinuities"] = "Not checked"

    else:
        route = np.round(route_raw).astype(np.int64)

        route_valid = (
            (route >= 0) &
            (route < len(arcs))
        )

        result["invalid_route_ids"] = int(
            (~route_valid).sum()
        )

        valid_route = route[route_valid]

        if len(valid_route) > 1:
            previous_ends = arcs[
                valid_route[:-1], 1
            ]

            next_starts = arcs[
                valid_route[1:], 0
            ]

            result["route_discontinuities"] = int(
                np.sum(previous_ends != next_starts)
            )
        else:
            result["route_discontinuities"] = 0

    return result


def main():
    all_results = []

    for partition in PARTITIONS:
        partition_path = DATASET_ROOT / partition

        if not partition_path.exists():
            print("Missing partition:", partition)
            continue

        folders = sorted(
            p for p in partition_path.iterdir()
            if p.is_dir()
        )

        print(
            f"\nAnalysing {partition}: {len(folders)} trips"
        )

        for index, folder in enumerate(folders, start=1):
            try:
                result = analyse_trip(folder, partition)
                all_results.append(result)

            except Exception as error:
                print(f"Error in {folder.name}: {error}")

            if index % 10 == 0:
                print(
                    f"Processed {index}/{len(folders)}"
                )

    if not all_results:
        print("No network results generated!")
        return

    output_file = (
        OUTPUT_DIR / "connectivity_report.csv"
    )

    fieldnames = sorted({
        key
        for result in all_results
        for key in result
    })

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== NETWORK SUMMARY ==========")

    for partition in PARTITIONS:
        rows = [
            r for r in all_results
            if r["partition"] == partition
        ]

        if not rows:
            continue

        print(f"\n{partition.upper()}")
        print("Trips analysed:", len(rows))

        print(
            "Total invalid arcs:",
            sum(r["invalid_arcs"] for r in rows)
        )

        print(
            "Total isolated nodes:",
            sum(r["isolated_nodes"] for r in rows)
        )

        print(
            "Average connected components:",
            round(
                np.mean([
                    r["connected_components"] for r in rows
                ]),
                2
            )
        )

        checked = [
            r for r in rows
            if isinstance(r["route_discontinuities"], int)
        ]

        print(
            "Trips with route discontinuities:",
            sum(
                r["route_discontinuities"] > 0
                for r in checked
            )
        )

        print(
            "Total route discontinuities:",
            sum(
                r["route_discontinuities"]
                for r in checked
            )
        )

    print("\nReport saved to:", output_file)


if __name__ == "__main__":
    main()