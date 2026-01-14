running = True

player_x = 10
player_y = 10

x_min = 0
x_max = 100
y_min = 0
y_max = 100

def scan_keys():
    inp = input("a/d/w/s to move, q to quit:")

    return inp

def render_state():
    print("player is at:", player_x, player_y)

def update_state(inp):
    global player_x, player_y, running
    if inp == "a":
        player_x -= 1
    elif inp == "d":
        player_x += 1
    elif inp == "w":
        player_y -= 1
    elif inp == "s":
        player_y += 1
    elif inp == "q":
        running = False

    if player_x < x_min:
        player_x = x_min
    if player_x > x_max:
        player_x = x_max
    if player_y < y_min:
        player_y = y_min
    if player_y > y_max:
        player_y = y_max

while running:
    # read/check for user actions (input)
    # update game state (physics, AI, etc)
    # render game state (graphics)

    render_state()
    inp = scan_keys()

    update_state(inp)