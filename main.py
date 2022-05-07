import cv2
import snake

# Process command line arguments
file_to_load = "Trapezium.jpg"
# Loads the desired image
image = cv2.imread( file_to_load, cv2.IMREAD_COLOR )
# Creates the snake
snake = snake.Snake( image, closed = True )
window_name = "Contouring"
# Core loop
while( True ):

    # Gets an image of the current state of the snake
    snakeImg = snake.viz()
    # Shows the image
    cv2.imshow( window_name, snakeImg )
    # Contouring snake step
    snake_changed = snake.step()
    # Stops looping when ESC pressed
    k = cv2.waitKey(33)
    if k == 27:
        break

cv2.destroyAllWindows()
