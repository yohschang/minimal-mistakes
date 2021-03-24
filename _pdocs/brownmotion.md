---
title: "Brownian motion"
permalink: /pdocs/brownmotion/
excerpt: "Brownian motion"
sitemap: true

sidebar:
  nav: "docs"

---

- This project is the simulation of Brownian motion
- Create particles class with different parameter represent by objects
```python
class Particle:
    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius
        self.mass = self.radius ** 2
    def x(self):
        return self.r[0]
    def x(self, value):
        self.r[0] = value
    def y(self):
        return self.r[1]
    def y(self, value):
        self.r[1] = value
    def vx(self):
        return self.v[0]
    def vx(self, value):
        self.v[0] = value
    def vy(self):
        return self.v[1]
    def vy(self, value):
        self.v[1] = value
    def overlaps(self, other):
        total = 0.
        dx = abs(self.x - other.x)
        dx = min(dx, 1-dx)
        dy = abs(self.y - other.y)
        dy = min(dy, 1-dy)
        return np.hypot(dx, dy) < self.radius + other.radius
```
- By creating numerous of partical with different radius the simple Brownian motion can be simulate
- Result

    - track one particle and plot on the left


![](https://i.imgur.com/ozgpWOf.gif)![](https://i.imgur.com/LqnoXut.png)
