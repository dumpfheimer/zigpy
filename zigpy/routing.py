"""Routing."""

import logging
import math

import zigpy.device
import zigpy.types as t
import zigpy.zdo.types as zdo_t

LOGGER = logging.getLogger(__name__)


def get_lqi(app: zigpy.application.ControllerApplication, source: t.EUI64,
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


def has_neighbor(app: zigpy.application.ControllerApplication, source: t.EUI64,
                 target: t.EUI64) -> bool | None:
    return get_lqi(app, source, target) is not None


def intersect_neighbors(list1: list[zdo_t.Neighbor], list2: list[zdo_t.Neighbor]) -> list[zdo_t.Neighbor]:
    ret: list[zdo_t.Neighbor] = []
    for n in list1:
        if list2.__contains__(n):
            ret.append(n)

    return ret


class DeviceRoute(zigpy.util.ListenableMixin):
    """A route to a device."""

    def __init__(self, app: zigpy.application.ControllerApplication,
                 dest: zigpy.device.Device,
                 hops: list[zdo_t.Neighbor]):
        self._app: zigpy.application.ControllerApplication = app
        self._dest: zigpy.deviec.Device = dest
        self._hops: list[zdo_t.Neighbor] = hops

        self._rating: float | None = None

    def _calculate_rating(self):
        rating: float = 0

        last_ieee = self._app.ieee

        for hop in self._hops:
            hop_lqi = get_lqi(self._app, last_ieee, hop.ieee)
            rating += math.pow(255-hop_lqi, 2)
            last_ieee = hop.ieee

        last_hop_lqi = get_lqi(self._app, last_ieee, self._dest.ieee)
        if last_hop_lqi is None:
            last_hop_lqi = 1024
        rating += math.pow(255-last_hop_lqi, 2)

        rating = math.sqrt(rating)
        rating = rating + len(self._hops) * 40
        self._rating = rating

        #route_str = "coordinator -> "
        #for hop in self._hops:
        #    route_str += str(hop.ieee) + " -> "
        #route_str += str(self._dest.ieee)
        #LOGGER.debug("Route %s rated with %.1f" % (route_str, rating))

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

    def __init__(self, app: zigpy.application.ControllerApplication,
                 dest: zigpy.device.Device):
        """Instantiate."""
        self._app: zigpy.application.ControllerApplication = app
        self._dest: zigpy.device.Device = dest

        self.possible_routes: list[DeviceRoute] | None = None

    def build(self):
        self.possible_routes = []

        reliable_lqi_threshold = 80 # lqi higher is better
        direct_lqi = get_lqi(self._app, self._app.ieee, self._dest.ieee)

        if direct_lqi is not None:
            #LOGGER.debug("routing.py: found direct link for %s (lqi %d)", str(self._dest.ieee), direct_lqi)
            self.possible_routes.append(DeviceRoute(self._app, self._dest, []))

        if direct_lqi is None or direct_lqi < reliable_lqi_threshold:
            #LOGGER.debug("routing.py: checking neighbors for %s", str(self._dest.ieee))
            neighbors_of_coordinator: list[zdo_t.Neighbor] = \
                self._app.topology.neighbors[
                    self._app.ieee]
            neighbors_of_target: list[zdo_t.Neighbor] = self._app.topology.neighbors[
                self._dest.ieee]

            def rating(route: DeviceRoute):
                return route.rating

            #LOGGER.debug("routing.py: checking neighbors for %s cp1", str(self._dest.ieee))
            # 1 hop routes
            shared_neighbors = intersect_neighbors(neighbors_of_target, neighbors_of_coordinator)
            #LOGGER.debug("routing.py: checking neighbors for %s cp2", str(self._dest.ieee))

            for one_hop in shared_neighbors:
                # only add links where link quality is better than direct
                if direct_lqi is None or one_hop.lqi > direct_lqi:
                    self.possible_routes.append(
                        DeviceRoute(self._app, self._dest, [one_hop]))

            #LOGGER.debug("routing.py: checking neighbors for %s cp3", str(self._dest.ieee))

            self.possible_routes.sort(key=rating)

            if len(self.possible_routes) > 0 and self.possible_routes[
                0].rating >= reliable_lqi_threshold:
                #LOGGER.debug("routing.py: happy with best one hop route for %s (%d)", str(self._dest.ieee), self.possible_routes[0].rating)
                return
            #if len(self.possible_routes) > 0:
            #    LOGGER.debug("routing.py: unhappy with best one hop route for %s (%d)", str(self._dest.ieee), self.possible_routes[0].rating)
            #else:
            #    LOGGER.debug(
            #        "routing.py: found no one hop route for %s", str(self._dest.ieee))

            # 2 hop routes
            for coordinator_hop in neighbors_of_coordinator:
                # only add links where link quality is better than direct
                if direct_lqi is None or coordinator_hop.lqi > direct_lqi:
                    for device_hop in neighbors_of_target:
                        if direct_lqi is None or device_hop.lqi > direct_lqi:
                            if has_neighbor(self._app, coordinator_hop.ieee, device_hop.ieee):
                                self.possible_routes.append(
                                    DeviceRoute(self._app, self._dest,
                                                [coordinator_hop, device_hop]))

            #LOGGER.debug("routing.py: sorting two hop routes %s", str(self._dest.ieee))
            self.possible_routes.sort(key=rating)

            #if len(self.possible_routes) > 0:
            #    LOGGER.debug("routing.py: best route with max 2 hops for %s: %d", str(self._dest.ieee), self.possible_routes[0].rating)
            #else:
            #    LOGGER.debug(
            #        "routing.py: found no route for %s", str(self._dest.ieee))

    def get_best_route(self) -> DeviceRoute | None:
        if self.possible_routes is None:
            self.build()
            #LOGGER.debug("Found %d routes for device %s", len(self.possible_routes), str(self._dest.ieee))
        return self.possible_routes[0] if len(self.possible_routes) > 0 else None
