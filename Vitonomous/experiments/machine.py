from typing import Any, Tuple


class Rectangle(object):

    def __init__(self):
        self.x: int = None
        self.y: int = None
        self.xy: Tuple[int, int] = None
        self.w: int = None
        self.h: int = None
        self.xx: int = None
        self.yy: int = None
        self.area: int = None

    def translate(self, x: int=None, y: int=None):
        x = 0 if x is None else x
        y = 0 if y is None else y
        if x == y == 0:
            return
        self.x += x
        self.y += y
        self.xx += x
        self.yy += y
        cx, cy = self.xy
        self.xy = (cx+x, cy+y)

    def move_to(self, x: int=None, y: int=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        self.translate(x - self.x, y - self.y)

    def distance_to(self, other: 'Rectangle'):
        return other.x - self.x, other.y - self.y

    def get_XY(self):
        return self.x, self.y

    def get_XXYY(self):
        return self.xx, self.yy

    def area_of(self, other: 'Rectangle'):
        return self.area / other.area

    def _initialise(self):
        self.area = self.w * self.h

    def _from_xy(self):
        cx, cy = self.xy
        self.x = cx - self.w//2
        self.y = cy - self.h//2
        self.xx = cx + self.w//2
        self.yy = cy + self.h//2

    def from_side_of(self, other: 'Rectangle', side: int, x_offset: float=None, w: int=None, h: int=None):
        w = other.w if w is None else w
        h = other.h if h is None else h
        x = other.x if side in [1, 3] else other.x-w if side == 4 else other.xx
        y = other.y if side in [2, 4] else other.y-h if side == 1 else other.yy
        x_offset = 0 if x_offset is None else x_offset
        x += int(other.w*x_offset-w//2)
        return self.from_x_y_w_h(x, y, w, h)

    def from_w_h(self, w: int, h: int):
        self.x = 0
        self.y = 0
        self.w = w
        self.h = h
        self.xy = (self.w//2, self.h//2)
        self.xx = self.w
        self.yy = self.h
        self._initialise()
        return self

    def from_x_y_w_h(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.xy = (self.x+self.w//2, self.y+self.h//2)
        self.xx = self.x+self.w
        self.yy = self.y+self.h
        self._initialise()
        return self

    def from_xy_s(self, xy: Tuple[int, int], s:int):
        self.xy = xy
        self.w = s
        self.h = self.w
        self._from_xy()
        self._initialise()
        return self

    def from_xy_w_h(self, xy: Tuple[int, int], w:int, h:int):
        self.xy = xy
        self.w = w
        self.h = h
        self._from_xy()
        self._initialise()
        return self

    def from_x_y_xx_yy(self, x: int, y: int, xx: int, yy: int):
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.w = self.xx - self.x
        self.h = self.yy - self.y
        self.xy = (self.x+self.w//2, self.y+self.h//2)
        self._initialise()
        return self
