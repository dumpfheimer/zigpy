"""Routing."""
import logging
import math

import zigpy.device
import zigpy.types as t
import zigpy.zdo.types as zdo_t

LOGGER = logging.getLogger(__name__)


def convert_lqi_to_rating(lqi):
    # will return a value from 0 (best, perfect) to 255 (bad)
    reliable_lqi_threshold = 100  # lqi higher is better
    if lqi <= reliable_lqi_threshold:
        return 255
    return 255 - (math.log10(lqi-reliable_lqi_threshold) / math.log10(255-reliable_lqi_threshold) * 255)


def get_lqi(app: zigpy.typing.ControllerApplicationType, source: t.EUI64,
            target: t.EUI64) -> t.uint8_t | None:
    if source in app.topology.neighbors:
        for neighbor in app.topology.neighbors[source]:
            if neighbor.ieee == target:
                return neighbor.lqi
    if target in app.topology.neighbors:
        for neighbor in app.topology.neighbors[target]:
            if neighbor.ieee == source:
                return neighbor.lqi
    return None


def has_neighbor(app: zigpy.typing.ControllerApplicationType, source: t.EUI64,
                 target: t.EUI64) -> bool | None:
    return get_lqi(app, source, target) is not None


def intersect_neighbors(list1: list[zdo_t.Neighbor], list2: list[zdo_t.Neighbor]) -> list[zdo_t.Neighbor]:
    ret: list[zdo_t.Neighbor] = []
    for n in list1:
        for n2 in list2:
            if n.ieee == n2.ieee:
                ret.append(n)

    return ret


class DeviceRoute(zigpy.util.ListenableMixin):
    """A route to a device."""

    def __init__(self, app: zigpy.typing.ControllerApplicationType,
                 dest: zigpy.device.Device,
                 hops: list[zdo_t.Neighbor]):
        self._app: zigpy.typing.ControllerApplicationType = app
        self._dest: zigpy.deviec.Device = dest
        self._hops: list[zdo_t.Neighbor] = hops

        self._rating: float | None = None
        self._debug_str: str | None = None

    def _calculate_rating(self):
        rating: float = 0

        route_str = "coordinator -> "

        last_ieee = self._app.state.node_info.ieee

        for hop in self._hops:
            hop_lqi = get_lqi(self._app, last_ieee, hop.ieee)
            rating += convert_lqi_to_rating(hop_lqi)
            last_ieee = hop.ieee
            route_str += str(hop.ieee) + " (lqi %d;%d) -> " % (hop_lqi, convert_lqi_to_rating(hop_lqi))

        last_hop_lqi = get_lqi(self._app, last_ieee, self._dest.ieee)
        if last_hop_lqi is None:
            last_hop_lqi = 1024
        rating += convert_lqi_to_rating(last_hop_lqi)
        route_str += str(self._dest.ieee) + " (lqi %d;%d). " % (last_hop_lqi, convert_lqi_to_rating(last_hop_lqi))

        #rating = rating + len(self._hops) * 30
        self._rating = rating
        route_str += " rating: %d" % self._rating
        self._debug_str = route_str

    def __str__(self):
        if self._debug_str is None:
            self._calculate_rating()
        return self._debug_str

    @property
    def rating(self):
        # lower is better
        if self._rating is None:
            self._calculate_rating()
        return self._rating

    def get_nwk_list(self) -> list[t.NWK]:
        ret: list[t.NWK] = []
        for hop in self._hops:
            ret.append(hop.nwk)

        return ret


class RouteBuilder(zigpy.util.ListenableMixin):
    """Find and rate routes to devices."""

    def __init__(self, app: zigpy.typing.ControllerApplicationType,
                 dest: zigpy.device.Device):
        """Instantiate."""
        self._app: zigpy.typing.ControllerApplicationType = app
        self._dest: zigpy.device.Device = dest

        self.possible_routes: list[DeviceRoute] | None = None

    def build(self) -> None:
        self.possible_routes = []

        reliable_lqi_threshold = 130 # lqi higher is better
        direct_lqi = get_lqi(self._app, self._app.state.node_info.ieee, self._dest.ieee)

        if direct_lqi is not None:
            LOGGER.debug("routing.py: found direct link for %s (lqi %d)", str(self._dest.ieee), direct_lqi)
            self.possible_routes.append(DeviceRoute(self._app, self._dest, []))

        if direct_lqi is None or direct_lqi < reliable_lqi_threshold:
            LOGGER.debug("routing.py: checking neighbors for %s", str(self._dest.ieee))
            neighbors_of_coordinator: list[zdo_t.Neighbor] = \
                self._app.topology.neighbors[
                    self._app.state.node_info.ieee]
            neighbors_of_target: list[zdo_t.Neighbor] = self._app.topology.neighbors[
                self._dest.ieee]

            def rating(route: DeviceRoute):
                return route.rating

            LOGGER.debug("routing.py: checking neighbors for %s cp1", str(self._dest.ieee))
            LOGGER.debug("routing.py: checking neighbors for %s cp1, have %d neighbors of coordinator" % (str(self._dest.ieee), len(neighbors_of_coordinator)))
            LOGGER.debug("routing.py: checking neighbors for %s cp1, have %d neighbors of target" % (str(self._dest.ieee), len(neighbors_of_target)))
            # 1 hop routes
            shared_neighbors = intersect_neighbors(neighbors_of_target, neighbors_of_coordinator)
            LOGGER.debug("routing.py: checking neighbors for %s cp2" % str(self._dest.ieee))
            LOGGER.debug("routing.py: checking neighbors for %s cp2, have %d shared neighbors" % (str(self._dest.ieee), len(shared_neighbors)))

            for one_hop in shared_neighbors:
                # only add links where link quality is better than direct
                if direct_lqi is None or one_hop.lqi > direct_lqi:
                    route = DeviceRoute(self._app, self._dest, [one_hop])
                    if route.rating <= 150:
                        LOGGER.debug("routing.py: using route for %s: %s" % (str(self._dest.ieee), route.__str__()))
                        self.possible_routes.append(route)
                    else:
                        LOGGER.debug("routing.py: discarding route for %s: %s" % (str(self._dest.ieee), route.__str__()))

            LOGGER.debug("routing.py: checking neighbors (%s) for %s cp3" % (str(len(shared_neighbors)), str(self._dest.ieee)))

            self.possible_routes.sort(key=rating)
            filter(lambda s: s.rating < 150, self.possible_routes)

            if len(self.possible_routes) > 0:
                if self.possible_routes[0].rating <= reliable_lqi_threshold * 2:
                    LOGGER.debug("routing.py: happy with best one hop route for %s (%d)" % (str(self._dest.ieee), self.possible_routes[0].rating))
                    return
                else:
                    LOGGER.debug("routing.py: best one hop route for %s has bad lqi: %d" % (str(self._dest.ieee), self.possible_routes[0].rating))
                    return

    def get_best_route(self) -> DeviceRoute | None:
        if self.possible_routes is None:
            self.build()
        if len(self.possible_routes):
            LOGGER.debug("Found %d routes for device %s" % (len(self.possible_routes), str(self._dest.ieee)))
            LOGGER.debug("best route for %s: %s" % (self._dest.ieee, self.possible_routes[0].__str__()))
        return self.possible_routes[0] if len(self.possible_routes) > 0 else None