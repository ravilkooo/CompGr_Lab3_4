import math
import random
import pygame
import os
import numpy as np

pygame.init()
W = 800
H = 800

sc = pygame.display.set_mode((W, H))
pygame.display.set_caption("CompGr_Lab3_4")

WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

FPS = 60  # число кадров в секунду
clock = pygame.time.Clock()


def rand_poly(min_rad, max_rad):
    M = random.randint(1, 6) % 6 + 3
    res = np.zeros((M, 4))
    angs = np.sort(np.random.rand(M) * 2 * math.pi)
    # while get_delta(angs) < 0.1: angs = np.sort(np.random.rand(M) * 2 * math.pi)
    rads = min_rad + (max_rad - min_rad) * np.sqrt(np.random.rand(M))
    for i in range(M):
        res[i][0] = int(rads[i] * np.cos(angs[i]))
        res[i][1] = int(rads[i] * np.sin(angs[i]))
        res[i][2] = 0
        res[i][3] = 1
    return res


def generate_prism(h, min_rad, max_rad):
    if min_rad > max_rad:
        min_rad, max_rad = max_rad, min_rad
    base1 = rand_poly(min_rad, max_rad)
    base2 = np.copy(base1)
    base2[:, 2] = h
    res = np.append(base1, base2, axis=0)
    # res = translation(res, [0, 0, max_rad])
    return res


def generate_cube(a):
    res = np.zeros((8, 4))
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                res[i * 2 * 2 + j * 2 + k][0] = (-1) ** k
                res[i * 2 * 2 + j * 2 + k][1] = (-1) ** j
                res[i * 2 * 2 + j * 2 + k][2] = (-1) ** i
    res[0], res[1] = np.copy(res[1]), np.copy(res[0])
    res[4], res[5] = np.copy(res[5]), np.copy(res[4])
    res = res * a
    res[:, 3] = 1
    # res = translation(res, [0, 0, a])
    return res


def get_c_m_2d(points):
    if points.shape[0] == 3:
        return np.mean(points, axis=0)
    z = np.mean(points, axis=0)
    s = np.sqrt(np.sum(np.cross(points[-1] - z, points[0] - z) ** 2)) / 2
    s_r = get_c_m_2d(np.array([z, points[0], points[-1]])) * s
    for i in range(points.shape[0] - 1):
        s_i = np.sqrt(np.sum(np.cross(points[i] - z, points[i + 1] - z) ** 2)) / 2
        s += s_i
        s_r += get_c_m_2d(np.array([z, points[i], points[i + 1]])) * s_i
    res = s_r / s
    return res


def get_c_m_3d(points):
    n = points.shape[0] // 2
    return (get_c_m_2d(points[0:n, 0:3]) + get_c_m_2d(points[n:, 0:3])) / 2


def draw_prism(prism):
    n = prism.shape[0] // 2
    pygame.draw.lines(sc, BLACK, True, [tp(p[0:2]) for p in prism][0:n])
    pygame.draw.lines(sc, BLACK, True, [tp(p[0:2]) for p in prism][n:])
    for i in range(n):
        pygame.draw.line(sc, BLACK, tp(prism[i, 0:2]), tp(prism[i + n, 0:2]))


def translation(p, dp):
    # print(f"Translation. dp = {dp[0]}, {dp[1]}, {dp[2]}")
    # print(f'Before: {p}')
    m = np.array([[1, 0, 0, dp[0]],
                  [0, 1, 0, dp[1]],
                  [0, 0, 1, dp[2]],
                  [0, 0, 0, 1]])
    # If lone point
    if len(p.shape) == 1:
        return m @ p
    # else (many points)
    res = np.zeros_like(p)
    for i in range(p.shape[0]):
        res[i] = m @ p[i]
    # print(f'After: {res}')
    return res


def rotation(p, a, c=np.array([0, 0, 0, 0]), axis=0):
    # print(f"Rotation. angle = {a}, center = ({c[0]}, {c[1]})")
    # x
    if axis == 0:
        m = np.array([[1, 0, 0, 0],
                      [0, np.cos(a), -np.sin(a), 0],
                      [0, np.sin(a), np.cos(a), 0],
                      [0, 0, 0, 1]])
    # y
    elif axis == 1:
        m = np.array([[np.cos(a), 0, np.sin(a), 0],
                      [0, 1, 0, 0],
                      [-np.sin(a), 0, np.cos(a), 0],
                      [0, 0, 0, 1]])
    # z
    else:
        m = np.array([[np.cos(a), -np.sin(a), 0, 0],
                      [np.sin(a), np.cos(a), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    # If lone point
    if len(p.shape) == 1:
        return translation(m @ translation(p, -c), c)
    # else (many points)
    res = np.zeros_like(p)
    for i in range(p.shape[0]):
        res[i] = translation(m @ translation(p[i], -c), c)
    return res


def tp(p):
    return np.array([W // 2 + p[0], H // 2 - p[1]])


def td(p):
    return p[0] - W // 2, H // 2 - p[1], 0


prism1 = generate_prism(60, 40, 80)
prism2 = generate_prism(60, 40, 80)
cube = generate_cube(40)
# cube = rotation(cube, math.pi / 6, get_c_m_3d(cube), axis=1)
prism1 = translation(prism1, [-160, 0, 0])
prism2 = translation(prism2, [160, 0, 0])


# prism2 = rotation(prism2, math.pi / 2, get_c_m_3d(prism2), axis=0)


def make_primary(points):
    return points


def make_isometric(points):
    res = rotation(points, math.pi, axis=1)
    res = rotation(res, np.arctan(1 / np.sqrt(2)), axis=0)
    return res


def make_perspective(points, d):
    res = translation(points, [0, 0, d])
    for p in res:
        w = d / p[2]
        p[0] *= w
        p[1] *= w
        p[2] = d
        p[3] = 1
    return res


mode = 0

anim = 0
anim_1 = 1
anim_2 = 1
anim_3 = 1
anim_4 = 1

while 1:
    # events handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                mode = 0
            if event.key == pygame.K_2:
                mode = 1
            if event.key == pygame.K_3:
                mode = 2
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                anim = 1 * (anim != 1)
                anim_1 = -anim_1 if (anim != 1) else anim_1
                continue
            elif event.key == pygame.K_F2:
                anim = 2 * (anim != 2)
                anim_2 = -anim_2 if (anim != 2) else anim_2
                continue
            elif event.key == pygame.K_F3:
                anim = 3 * (anim != 3)
                anim_3 = -anim_3 if (anim != 3) else anim_3
                continue
            elif event.key == pygame.K_F4:
                anim = 4 * (anim != 4)
                anim_4 = -anim_4 if (anim != 4) else anim_4
                continue
        if anim != 0:
            continue

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                cube = rotation(cube, math.pi / 10, get_c_m_3d(cube), axis=0)
                prism1 = rotation(prism1, math.pi / 10, get_c_m_3d(prism1), axis=1)
                prism2 = rotation(prism2, math.pi / 10, get_c_m_3d(prism2), axis=2)
            if event.key == pygame.K_F5:
                prism1 = generate_prism(60, 40, 80)
                prism2 = generate_prism(60, 40, 80)
                cube = generate_cube(40)
                prism1 = translation(prism1, [-160, 0, 0])
                prism2 = translation(prism2, [160, 0, 0])
            if event.key == pygame.K_F6:
                cube = generate_cube(40)
                prism1 = generate_prism(60, 20, 40)
                prism2 = generate_prism(60, 10, 20)
                prism1 = translation(prism1, [-120, 0, 0])
                prism2 = translation(prism2, [-150, 0, 0])

    # net
    sc.fill(WHITE)
    vert = np.append(np.arange(W / 2, 0, -20), np.arange(W / 2, W, 20))
    horiz = np.append(np.arange(H / 2, 0, -20), np.arange(H / 2, H, 20))
    for v_line in vert:
        pygame.draw.line(sc, GREY, (v_line, 0), (v_line, H))
    for h_line in horiz:
        pygame.draw.line(sc, GREY, (0, h_line), (W, h_line))
    pygame.draw.line(sc, BLACK, (W / 2, 0), (W / 2, H))
    pygame.draw.line(sc, BLACK, (0, H / 2), (W, H / 2))

    # animations
    if anim == 1:
        cube = translation(cube, [anim_1, 0, 0])
        prism1 = translation(prism1, [0, anim_1, 0])
        prism2 = translation(prism2, [0, -anim_1, 0])
    elif anim == 2:
        cube = rotation(cube, 0.01 * anim_2, get_c_m_3d(cube), axis=1)
        prism1 = rotation(prism1, 0.01 * anim_2, get_c_m_3d(prism1), axis=1)
        prism2 = rotation(prism2, 0.01 * anim_2, get_c_m_3d(prism2), axis=1)
    elif anim == 3:
        cube = rotation(cube, 0.01 * anim_3, get_c_m_3d(cube), axis=1)
        prism1 = rotation(prism1, 0.01 * anim_3, get_c_m_3d(prism1), axis=2)
        prism1 = rotation(prism1, 0.01 * anim_3, get_c_m_3d(cube), axis=1)
        prism2 = rotation(prism2, 0.01 * anim_3, get_c_m_3d(prism2), axis=2)
        prism2 = rotation(prism2, 0.01 * anim_3, get_c_m_3d(cube), axis=2)
    elif anim == 4:
        cube = rotation(cube, 0.01 * anim_4, get_c_m_3d(cube), axis=1)
        cube = rotation(cube, 0.01 * anim_4, get_c_m_3d(cube), axis=2)
        prism1 = rotation(prism1, 0.01 * anim_4, get_c_m_3d(prism1), axis=2)
        prism1 = rotation(prism1, 0.01 * anim_4, get_c_m_3d(cube), axis=1)
        prism2 = rotation(prism2, 0.01 * anim_4, get_c_m_3d(prism2), axis=2)
        prism2 = rotation(prism2, 0.01 * anim_4, get_c_m_3d(prism1), axis=2)

    # drawing
    if mode == 0:
        draw_prism(cube)
        draw_prism(prism1)
        draw_prism(prism2)
    if mode == 1:
        draw_prism(make_isometric(cube))
        draw_prism(make_isometric(prism1))
        draw_prism(make_isometric(prism2))
    if mode == 2:
        draw_prism(make_perspective(cube, 300))
        draw_prism(make_perspective(prism1, 300))
        draw_prism(make_perspective(prism2, 300))

    # tick
    pygame.display.update()
    clock.tick(FPS)
