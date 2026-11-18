"""SUMO model of the intersection: network (netconvert), signal programs and demand.

Lane numbering: SUMO lane 0 of every edge is the sidewalk, so our lane 1 (curbside) is SUMO lane 1.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tlopt.intersection import PERPENDICULAR, Intersection

APPROACH_M = 300.0
LANE_W, WALK_W = 3.5, 2.0
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
RIGHT_OF = {"N": "W", "S": "E", "E": "N", "W": "S"}   # where traffic coming FROM there turns right to
LEFT_OF = {"N": "E", "S": "W", "E": "S", "W": "N"}
POSITION = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
VTYPES = {   # (vClass, length m, accel, decel, HBEFA4 emission class)
    "car": ("passenger", 4.5, 2.6, 4.5, "HBEFA4/PC_petrol_Euro-6ab"),
    "truck": ("delivery", 6.5, 1.8, 4.0, "HBEFA4/RT_le7.5t_Euro-VI_A-C"),   # pickups and box trucks
    "bus": ("bus", 12.0, 1.2, 4.0, "HBEFA4/UBus_Std_gt15-18t_Euro-VI_A-C"),
    "motorcycle": ("motorcycle", 2.2, 3.0, 5.0, "HBEFA4/MC_4S_gt250cc_Euro-5"),
}


def sumo_home() -> Path:
    import sumo
    return Path(sumo.SUMO_HOME)


def binary(name: str) -> str:
    path = sumo_home() / "bin" / (name + (".exe" if os.name == "nt" else ""))
    return str(path)


def sumo_env() -> dict:
    return {**os.environ, "SUMO_HOME": str(sumo_home())}


def work_dir() -> Path:
    """A short temporary folder (Windows has a 260-character path limit)."""
    if os.name == "nt":
        base = Path(os.environ.get("SystemDrive", "C:") + "\\") / "tlopt_tmp"
        try:
            base.mkdir(exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="r", dir=base))
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="tlopt_"))


def exit_of(approach: str, movement: str) -> str:
    return {"T": OPPOSITE[approach], "R": RIGHT_OF[approach], "L": LEFT_OF[approach]}[movement]


def build_network(inter: Intersection, folder: Path) -> Path:
    """Write nodes, edges and connections and run netconvert. Returns the .net.xml path."""
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="C", x="0", y="0", type="traffic_light", tlType="static")
    edges, cons = ET.Element("edges"), ET.Element("connections")
    for leg in inter.approaches:
        dx, dy = POSITION[leg]
        ET.SubElement(nodes, "node", id=leg, x=str(dx * APPROACH_M), y=str(dy * APPROACH_M))
    lanes_out = {leg: len(inter.approaches[leg].lanes) for leg in inter.approaches}
    for leg, geo in inter.approaches.items():
        speed = geo.speed_limit_mph * 0.44704
        for edge_id, src, dst, count in ((f"{leg}_in", leg, "C", len(geo.lanes)), (f"{leg}_out", "C", leg, lanes_out[leg])):
            edge = ET.SubElement(edges, "edge", id=edge_id, attrib={"from": src, "to": dst}, numLanes=str(count + 1),
                                 speed=f"{speed:.2f}", priority="1")
            ET.SubElement(edge, "lane", index="0", allow="pedestrian", width=str(WALK_W))
            for index in range(1, count + 1):
                ET.SubElement(edge, "lane", index=str(index), disallow="pedestrian", width=str(LANE_W))
        for lane, moves in geo.lanes.items():
            for move in moves:
                target = exit_of(leg, move)
                if target not in inter.approaches:
                    continue
                n_out = lanes_out[target]
                to_lane = 1 if move == "R" else n_out if move == "L" else min(int(lane), n_out)
                ET.SubElement(cons, "connection", attrib={"from": f"{leg}_in", "to": f"{target}_out",
                                                         "fromLane": lane, "toLane": str(to_lane)})
        ET.SubElement(cons, "crossing", node="C", edges=f"{leg}_in {leg}_out", priority="false", width="3")
    for name, root in (("n.nod.xml", nodes), ("n.edg.xml", edges), ("n.con.xml", cons)):
        ET.ElementTree(root).write(folder / name)
    net = folder / "n.net.xml"
    result = subprocess.run([binary("netconvert"), "-n", "n.nod.xml", "-e", "n.edg.xml", "-x", "n.con.xml",
                             "-o", net.name, "--no-turnarounds", "true", "--walkingareas", "true",
                             "--junctions.corner-detail", "5"],
                            cwd=folder, capture_output=True, text=True, env=sumo_env())
    if result.returncode != 0 or not net.exists():
        raise RuntimeError(f"netconvert failed: {result.stderr[-2000:]}")
    return net


@dataclass
class Link:
    index: int
    approach: str | None    # vehicle link: the approach it comes from
    movement: str | None
    crossing_leg: str | None   # pedestrian crossing link: the leg it crosses


def signal_links(net: Path) -> list[Link]:
    """Which signal link index controls which movement or crosswalk (read from the .net.xml)."""
    root = ET.parse(net).getroot()
    crossing_leg = {e.get("id"): e.get("crossingEdges").split()[0].split("_")[0]
                    for e in root.iter("edge") if e.get("function") == "crossing"}
    links: dict[int, Link] = {}
    for con in root.iter("connection"):
        if con.get("tl") != "C":
            continue
        index = int(con.get("linkIndex"))
        src, dst = con.get("from"), con.get("to")
        if dst in crossing_leg:
            links[index] = Link(index, None, None, crossing_leg[dst])
        elif src.endswith("_in"):
            approach, target = src[0], dst[0]
            move = next(m for m in "TRL" if exit_of(approach, m) == target)
            links[index] = Link(index, approach, move, None)
    return [links[i] for i in sorted(links)]


def _side_len(inter: Intersection, plan: dict, side: int) -> float:
    rings = inter.ring_sides()[side]
    return max(sum(sum(plan["phases"][str(q)][k] for k in ("green_s", "yellow_s", "all_red_s")) for q in ring)
               for ring in rings if ring)


def timeline(inter: Intersection, plan: dict) -> list[tuple[float, float, int, str]]:
    """Both rings laid out over one cycle: (start, end, phase, 'G'|'y'|'r'). Each barrier side
    starts when the previous one ends; a ring that is shorter on a side waits in red."""
    out, side_start = [], 0.0
    for side, rings in enumerate(inter.ring_sides()):
        for ring in rings:
            t = side_start
            for p in ring:
                spec = plan["phases"][str(p)]
                for colour, duration in (("G", spec["green_s"]), ("y", spec["yellow_s"]), ("r", spec["all_red_s"])):
                    out.append((t, t + duration, p, colour))
                    t += duration
        side_start += _side_len(inter, plan, side)
    return out


def tls_program(inter: Intersection, plan: dict, links: list[Link], program_id: str) -> ET.Element:
    """SUMO tlLogic for a plan: greens, yellows and all-reds of both rings, in NEMA order."""
    segments = timeline(inter, plan)
    cycle = sum(_side_len(inter, plan, s) for s in range(len(inter.ring_sides())))
    walk = inter.pedestrians.walk_s
    cuts = {0.0, round(cycle, 1)} | {round(x, 1) for s in segments for x in s[:2]}
    cuts |= {round(min(s[0] + walk, s[1]), 1) for s in segments if s[3] == "G"}   # end of WALK
    cuts = sorted(c for c in cuts if c <= cycle)
    states: list[list] = []
    for start, end in zip(cuts, cuts[1:]):
        mid = (start + end) / 2
        active = {s[2]: (s[3], s[0]) for s in segments if s[0] <= mid < s[1]}
        state = "".join(_link_state(inter, link, active, mid, walk) for link in links)
        if states and states[-1][0] == state:
            states[-1][1] += end - start
        else:
            states.append([state, end - start])
    logic = ET.Element("tlLogic", id="C", type="static", programID=program_id, offset="0")
    for state, duration in states:
        ET.SubElement(logic, "phase", duration=f"{duration:.1f}", state=state)
    return logic


def _link_state(inter: Intersection, link: Link, active: dict, t: float, walk: float) -> str:
    if link.crossing_leg is not None:
        for p, (colour, start) in active.items():
            phase = inter.phases[str(p)]
            if colour == "G" and "T" in phase.movements and phase.approach in PERPENDICULAR[link.crossing_leg] \
                    and t < start + walk:
                return "G"
        return "r"
    best = "r"
    for p, (colour, _start) in active.items():
        phase = inter.phases[str(p)]
        if phase.approach != link.approach or link.movement not in phase.movements:
            continue
        if colour == "G":
            minor = link.movement == "R" or (link.movement == "L" and phase.left != "protected")
            best = "G" if not minor else ("g" if best != "G" else best)
        elif colour == "y" and best == "r":
            best = "y"
    return best


def write_additional(inter: Intersection, plans: dict[str, dict], links: list[Link], net: Path,
                     folder: Path, detector_period: float) -> Path:
    root = ET.Element("additional")
    for name, plan in plans.items():
        root.append(tls_program(inter, plan, links, name))
    for leg, geo in inter.approaches.items():
        for lane in geo.lanes:
            ET.SubElement(root, "laneAreaDetector", id=f"q_{leg}_{lane}", lane=f"{leg}_in_{lane}", pos="0",
                          endPos="-0.1", friendlyPos="true", period=f"{detector_period:.1f}", file="queues.xml")
    path = folder / "plans.add.xml"
    ET.ElementTree(root).write(path)
    return path


def write_routes(inter: Intersection, demand: dict, folder: Path, seed: int, warmup_s: float,
                 tau: float, name: str = "r.rou.xml") -> Path:
    """demand: {(interval_index, approach, movement, vtype): mean vehicles in that 15 min}.
    Interval -1 is the warm-up. Poisson count per interval, departures spread uniformly."""
    rng = np.random.default_rng(seed)
    root = ET.Element("routes")
    for vtype, (vclass, length, accel, decel, emission) in VTYPES.items():
        ET.SubElement(root, "vType", id=vtype, vClass=vclass, length=str(length), accel=str(accel),
                      decel=str(decel), sigma="0.5", tau=f"{tau:.3f}", carFollowModel="Krauss",
                      emissionClass=emission, minGap="2.0")
    vehicles = []
    for (interval, approach, movement, vtype), mean in sorted(demand.items()):
        start = warmup_s + interval * 900.0
        for k, depart in enumerate(np.sort(rng.uniform(start, start + 900.0, rng.poisson(mean)))):
            vehicles.append((float(depart), f"{approach}{movement}_{vtype}_{interval + 1}_{k}", vtype,
                             f"{approach}_in {exit_of(approach, movement)}_out"))
    for depart, vid, vtype, edges in sorted(vehicles):
        car = ET.SubElement(root, "vehicle", id=vid, type=vtype, depart=f"{depart:.1f}", departLane="best",
                            departSpeed="max")
        ET.SubElement(car, "route", edges=edges)
    end = warmup_s + 900.0 * (1 + max((k[0] for k in demand), default=0))
    for leg, rate in inter.pedestrians.peds_per_hour.items():
        if leg in inter.approaches and rate > 0:
            for a, b in ((f"{leg}_in", f"{leg}_out"), (f"{leg}_out", f"{leg}_in")):
                flow = ET.SubElement(root, "personFlow", id=f"ped_{a}", begin="0", end=f"{end:.0f}",
                                     probability=f"{rate / 2 / 3600:.6f}",
                                     departPos=str(APPROACH_M - 15 if a.endswith("_in") else 5))
                ET.SubElement(flow, "walk", attrib={"from": a, "to": b,
                                                    "arrivalPos": str(5 if b.endswith("_out") else APPROACH_M - 15)})
    path = folder / name
    ET.ElementTree(root).write(path)
    return path
