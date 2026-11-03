

class Ground:
    """Conservative upright robot envelope tested against actual PhysX colliders."""

    def __init__(self, rt, out):
        import omni.physx
        self.q = omni.physx.get_physx_scene_query_interface()
        self.cache = {}
        self.out = out

    def free(self, x, y):
        key = (round(float(x), 4), round(float(y), 4))
        if key in self.cache:
            return self.cache[key]
        hits = []

        def hit(h):
            p = str(h.rigid_body) + str(h.collision)
            if '/Robot/' not in p and (not p.endswith('/Robot')):
                hits.append(p)
            return True
        self.q.overlap_box((0.35, 0.35, 0.85), (key[0], key[1], 0.97), (0.0, 0.0, 0.0, 1.0), hit, False)
        floor = self.q.raycast_closest((key[0], key[1], 0.1), (0.0, 0.0, -1.0), 0.3)
        valid = not hits and bool(floor.get('hit')) and (abs(float(floor.get('position', [0, 0, 99])[2])) < 0.08) and (float(floor.get('normal', [0, 0, 0])[2]) > 0.95)
        self.cache[key] = {'legal': valid, 'obstacles': hits[:8], 'floor_hit': bool(floor.get('hit')), 'floor_z': float(floor['position'][2]) if floor.get('hit') else None}
        return self.cache[key]
