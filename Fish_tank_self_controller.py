import pygame
import random
import sys
import math
import socket
# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BLUE = (0, 100, 255)
DARK_BLUE = (0, 50, 150)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (150, 50, 255)

# Game settings - initial values
BUBBLE_SPEED = 2
BUBBLE_SPAWN_RATE = 0.02
BUBBLE_SIZE = 20
OBSTACLE_WIDTH = 80

class VerticalController:
    """Provides automated vertical movement with speed and direction"""
    def __init__(self):
        self.current_speed = 2.0          # Current movement speed
        self.current_direction = 0.0      # 0 = neutral, 1 = down, -1 = up
        self.target_speed = 2.0           # Target to smoothly transition to
        self.target_direction = 0.0       # Target direction
        self.change_interval = 0.1        # Time in seconds between changes
        self.last_change_time = 0         # Time tracker
        self.transition_speed = 1      # How quickly to move toward targets (0-1)
        self.speed_range = (1.0, 5.0)     # Min/max speed
        self.manual_override = False      # Flag for when player takes control
        self.manual_duration = 0          # How long manual control lasts
        
    def update(self, elapsed_time,y_axis):
        """Update speed and direction based on elapsed time"""
        # If manual override is active
        if self.manual_override:
            self.manual_duration -= 1
            if self.manual_duration <= 0:
                self.manual_override = False
            return self.current_speed, self.current_direction
            
        # Check if it's time to change targets
        if elapsed_time - self.last_change_time >= self.change_interval:
            # Set new random targets
            #self.target_speed = random.uniform(self.speed_range[0], self.speed_range[1])
            # Direction is now -1 (up), 0 (neutral), or 1 (down)
            #self.target_direction = random.choice([-1, -0.5, 0, 0.5, 1])
            self.target_speed=5*(abs(y_axis))                        #modify
            self.target_direction=-int(y_axis > 0) + int(y_axis < 0)      #-1 if y_axis>0, 1 if y_axis<0, 0 if 0
            self.last_change_time = elapsed_time
        
        # Smoothly interpolate current values toward targets
        # Speed interpolation
        speed_diff = self.target_speed - self.current_speed
        self.current_speed += speed_diff * self.transition_speed
        
        # Direction interpolation
        dir_diff = self.target_direction - self.current_direction            
        #self.current_direction += dir_diff * self.transition_speed           #modify
        self.current_direction=self.target_direction
        
        # Clamp direction to valid range
        self.current_direction = max(-1, min(1, self.current_direction))
        
        return self.current_speed, self.current_direction
    
    def set_manual_speed(self, speed):
        """Override with manual speed setting"""
        self.current_speed = speed
        self.manual_override = True
        self.manual_duration = 30  # Override for half a second (30 frames at 60fps)
    
    def set_manual_direction(self, direction):
        """Override with manual direction setting (-1=up, 0=neutral, 1=down)"""
        self.current_direction = direction
        self.manual_override = True
        self.manual_duration = 60  # Override for 1 second (60 frames at 60fps)

class Player:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.width = 60
        self.height = 30
        self.y_speed = 0
        self.max_speed = 8
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
    def update(self, speed, direction):
        # direction: -1 = up, 0 = neutral, 1 = down
        # Convert to actual y speed
        self.y_speed = direction * speed
        
        # Update position
        self.y += self.y_speed
        
        # Keep player on screen
        if self.y < 0:
            self.y = 0
        elif self.y > SCREEN_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - self.height
            
        # Update rect for collision detection
        self.rect.y = self.y
        
    def draw(self, screen):
        # Draw fish body
        fish_color = ORANGE  # Orange
        pygame.draw.ellipse(screen, fish_color, (self.x, self.y, self.width, self.height))
        
        # Draw tail
        tail_points = [
            (self.x, self.y + self.height // 2),
            (self.x - 20, self.y),
            (self.x - 20, self.y + self.height)
        ]
        pygame.draw.polygon(screen, fish_color, tail_points)
        
        # Draw eye
        pygame.draw.circle(screen, WHITE, (self.x + self.width - 15, self.y + 10), 8)
        pygame.draw.circle(screen, (0, 0, 0), (self.x + self.width - 12, self.y + 10), 4)
        
class Bubble:
    def __init__(self):
        self.size = random.randint(5, BUBBLE_SIZE)
        self.x = SCREEN_WIDTH + self.size
        self.y = random.randint(self.size, SCREEN_HEIGHT - self.size)
        self.speed = BUBBLE_SPEED * (0.5 + random.random())
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        
    def update(self):
        self.x -= self.speed
        self.rect.x = self.x
        
    def draw(self, screen):
        # Grey bubble with a lighter highlight
        bubble_color = (180, 180, 180)  # Grey color
        pygame.draw.circle(screen, bubble_color, (self.x, self.y), self.size)
        # Add a little reflection highlight
        highlight_color = (220, 220, 220)  # Light grey for highlight
        pygame.draw.circle(screen, highlight_color, (self.x - self.size//3, self.y - self.size//3), self.size//4)

class Obstacle:
    def __init__(self, obstacle_speed):
        self.width = OBSTACLE_WIDTH
        self.height = random.randint(100, 300)
        self.x = SCREEN_WIDTH + self.width
        # Randomly place obstacle at top or bottom
        self.is_top = random.choice([True, False])
        if self.is_top:
            self.y = 0
        else:
            self.y = SCREEN_HEIGHT - self.height
        self.speed = obstacle_speed
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.color = DARK_BLUE
        
    def update(self):
        self.x -= self.speed
        self.rect.x = self.x
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # Add some detail to the obstacle (like coral or seaweed)
        if self.is_top:
            for i in range(3):
                x_pos = self.x + i * (self.width // 3) + 10
                weed_height = random.randint(20, 50)
                pygame.draw.line(screen, GREEN, (x_pos, self.height), 
                                (x_pos + random.randint(-10, 10), self.height - weed_height), 5)
        else:
            for i in range(3):
                x_pos = self.x + i * (self.width // 3) + 10
                pygame.draw.circle(screen, (255, 100, 100), 
                                  (x_pos, self.y + random.randint(10, 30)), 10)

def generate_bubbles(bubbles, spawn_rate):
    if random.random() < spawn_rate:
        bubbles.append(Bubble())
    return bubbles

def generate_obstacles(obstacles, spawn_rate, obstacle_speed):
    # Check if we should spawn a new obstacle
    if random.random() < spawn_rate:
        # Check if there's enough distance from the last obstacle
        min_distance = 250  # Minimum pixel gap between obstacles
        can_spawn = True
        
        # Find the rightmost obstacle (closest to screen edge)
        rightmost_x = 0
        for obstacle in obstacles:
            if obstacle.x > rightmost_x:
                rightmost_x = obstacle.x
        
        # Only spawn if there's enough distance or no obstacles
        if obstacles and (SCREEN_WIDTH - rightmost_x) < min_distance:
            can_spawn = False
            
        if can_spawn:
            obstacles.append(Obstacle(obstacle_speed))
    
    return obstacles

def main():
    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fish Tank Game - Self Controller")
    clock = pygame.time.Clock()
    
    # Game objects
    player = Player()
    controller = VerticalController()
    bubbles = []
    obstacles = []
    score = 0
    game_over = False
    elapsed_time = 0
    
    # Game difficulty variables - these can be modified inside the function
    obstacle_spawn_rate = 0.01
    obstacle_speed = 3
    
    # Direction control buttons
    direction_buttons = []
    directions = ["Up", "Stop", "Down"]
    direction_values = [-1, 0, 1]
    button_width = 80
    button_height = 40
    button_y = 15

    server = None
    client = None
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 5000))
        server.listen(1)
        #server.settimeout(10)  # Non-blocking with short timeout
        
        print("Waiting for Android to connect...")
        client, addr = server.accept()
        client.settimeout(0.01)  # Make receive non-blocking with short timeout
        print(f"Connected to {addr}")
    except Exception as e:
        print(f"Socket error: {e}")
        pygame.quit()
        sys.exit()
    
    for i, (label, value) in enumerate(zip(directions, direction_values)):
        x = 200 + i * (button_width + 10)
        direction_buttons.append({
            'rect': pygame.Rect(x, button_y, button_width, button_height),
            'label': label,
            'value': value
        })
    
    # Speed control buttons
    speed_buttons = []
    speed_labels = ["Slow", "Medium", "Fast"]
    speed_values = [1.0, 3.0, 5.0]
    
    for i, (label, value) in enumerate(zip(speed_labels, speed_values)):
        x = 500 + i * (button_width + 10)
        speed_buttons.append({
            'rect': pygame.Rect(x, button_y, button_width, button_height),
            'label': label,
            'value': value
        })
    
    # Game loop
    running = True

    loopRunning = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if game_over and event.key == pygame.K_SPACE:
                    # Restart game
                    player = Player()
                    controller = VerticalController()
                    bubbles = []
                    obstacles = []
                    score = 0
                    game_over = False
                    obstacle_spawn_rate = 0.01  # Reset difficulty
                    obstacle_speed = 3
                    elapsed_time = 0
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check direction buttons
                for btn in direction_buttons:
                    if btn['rect'].collidepoint(event.pos):
                        controller.set_manual_direction(btn['value'])
                
                # Check speed buttons
                for btn in speed_buttons:
                    if btn['rect'].collidepoint(event.pos):
                        controller.set_manual_speed(btn['value'])
        
        # Update elapsed time (in seconds)
        elapsed_time += 1/60  # Assuming 60 FPS
        
        if not game_over:
            # Get control values from the controller
            #y_axis=random.uniform(-1,1)
            y_axis = 0  # Default value if no data received
            
            try:
                data = client.recv(1024).decode().strip()
                if data:  # Only process if we actually got data
                    try:
                        y_axis = float(data)
                    except ValueError:
                        print(f"Received invalid data: {data}")
            except socket.timeout:
                # This is expected behavior for non-blocking sockets
                pass
            except Exception as e:
                print(f"Socket error during receive: {e}")
            speed, direction = controller.update(elapsed_time,y_axis)
            
            # Update player
            player.update(speed, direction)
            
            # Generate and update bubbles
            bubbles = generate_bubbles(bubbles, BUBBLE_SPAWN_RATE)
            bubbles = [bubble for bubble in bubbles if bubble.x + bubble.size > 0]
            for bubble in bubbles:
                bubble.update()
                
            # Generate and update obstacles
            obstacles = generate_obstacles(obstacles, obstacle_spawn_rate, obstacle_speed)
            obstacles = [obstacle for obstacle in obstacles if obstacle.x + obstacle.width > 0]
            for obstacle in obstacles:
                obstacle.update()
            
            # Collision detection
            for bubble in bubbles[:]:
                if player.rect.colliderect(bubble.rect):
                    bubbles.remove(bubble)
                    score += 1
                    
            for obstacle in obstacles:
                if player.rect.colliderect(obstacle.rect):
                    game_over = True
                    
            # Increase difficulty over time
            if score > 0 and score % 10 == 0:
                obstacle_spawn_rate = min(0.05, obstacle_spawn_rate + 0.001)
                obstacle_speed = min(8, obstacle_speed + 0.1)
        
        # Drawing
        # Draw background - just a solid blue
        screen.fill(BLUE)
        
        # Draw game objects
        for bubble in bubbles:
            bubble.draw(screen)
            
        for obstacle in obstacles:
            obstacle.draw(screen)
            
        player.draw(screen)
        
        # Draw score
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw control panel
        # Draw direction buttons
        for btn in direction_buttons:
            # Highlight active direction
            color = PURPLE if abs(btn['value'] - controller.current_direction) < 0.5 else (100, 100, 100)
            pygame.draw.rect(screen, color, btn['rect'])
            text = pygame.font.SysFont(None, 24).render(btn['label'], True, WHITE)
            text_rect = text.get_rect(center=btn['rect'].center)
            screen.blit(text, text_rect)
        
        # Draw speed buttons
        for btn in speed_buttons:
            # Highlight active speed range
            color = GREEN if abs(btn['value'] - controller.current_speed) < 1.0 else (100, 100, 100)
            pygame.draw.rect(screen, color, btn['rect'])
            text = pygame.font.SysFont(None, 24).render(btn['label'], True, WHITE)
            text_rect = text.get_rect(center=btn['rect'].center)
            screen.blit(text, text_rect)
        
        # Draw current controller values
        dir_text = "Neutral"
        if controller.current_direction < -0.3:
            dir_text = "Up"
        elif controller.current_direction > 0.3:
            dir_text = "Down"
            
        speed_text = font.render(f"Speed: {controller.current_speed:.1f}", True, WHITE)
        dir_text = font.render(f"Direction: {dir_text}", True, WHITE)
        screen.blit(speed_text, (10, 50))
        screen.blit(dir_text, (10, 90))
        
        # Draw game over
        if game_over:
            font = pygame.font.SysFont(None, 72)
            game_over_text = font.render("GAME OVER", True, RED)
            screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, 
                                         SCREEN_HEIGHT//2 - game_over_text.get_height()//2))
            
            font = pygame.font.SysFont(None, 36)
            restart_text = font.render("Press SPACE to restart", True, YELLOW)
            screen.blit(restart_text, (SCREEN_WIDTH//2 - restart_text.get_width()//2, 
                                      SCREEN_HEIGHT//2 + 50))
        
        # Draw control indicator - shows current direction from controller
        indicator_height = 200
        indicator_y = SCREEN_HEIGHT - indicator_height - 20
        pygame.draw.rect(screen, (50, 50, 50), (SCREEN_WIDTH - 50, indicator_y, 30, indicator_height))
        
        # Calculate position based on controller direction (-1 to 1)
        normalized_pos = (controller.current_direction + 1) / 2  # Convert to 0-1 range
        position_y = indicator_y + (indicator_height * normalized_pos)
        pygame.draw.circle(screen, RED, (SCREEN_WIDTH - 35, int(position_y)), 12)
        
        # Draw labels
        up_text = pygame.font.SysFont(None, 24).render("UP", True, WHITE)
        down_text = pygame.font.SysFont(None, 24).render("DOWN", True, WHITE)
        screen.blit(up_text, (SCREEN_WIDTH - 80, indicator_y - 5))
        screen.blit(down_text, (SCREEN_WIDTH - 95, indicator_y + indicator_height - 5))
        
        # Update display
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
