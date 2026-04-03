import pygame

pygame.init()
pygame.joystick.init()

js = pygame.joystick.Joystick(0)
js.init()

while True:
    pygame.event.pump()

    lx = js.get_axis(0)   # 左摇杆X
    ly = js.get_axis(1)   # 左摇杆Y
    rx = js.get_axis(3)   # 右摇杆X

    print(f"lx={lx:.2f}, ly={ly:.2f}, rx={rx:.2f}")