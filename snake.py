import numpy as np
import cv2
import math


class Snake:
    """ A class for active contour using Snakes """
    width = -1          # The image width.
    height = -1         # The image height.
    points = None       # The list of points of the snake.
    n_starting_points = 50       # The number of starting points of the snake.
    snake_length = 0    # The length of the snake (euclidean distances).
    closed = True       # Indicates if the snake is closed or open.
    alpha = 0.5         # The weight of the uniform energy.
    beta = 0.5          # The weight of the curvture energy.
    image = None        # The source image.
    gray = None         # The image in grayscale.
    binary = None       # The image in binary (threshold method).
    gradientX = None    # The gradient (sobel) of the image relative to x.
    gradientY = None    # The gradient (sobel) of the image relative to y.
    blur = None
    MinD = 5    # The minimum distance between two points to consider them overlaped
    MaxD = 50    # The maximum distance to insert another point into the spline
    sks = 7              # The size of the search kernel.
    
    def __init__( self, image = None, closed = True ):

        # Sets the image and it's properties
        self.image = image

        # Image properties
        self.width = image.shape[1]
        self.height = image.shape[0]

        # Image variations used by the snake
        self.gray = cv2.cvtColor( self.image, cv2.COLOR_RGB2GRAY )
        self.binary = cv2.adaptiveThreshold( self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2 )
        self.gradientX = cv2.Sobel( self.gray, cv2.CV_64F, 1, 0, ksize=5 )
        self.gradientY = cv2.Sobel( self.gray, cv2.CV_64F, 0, 1, ksize=5 )
        self.closed = closed

        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )

        if self.closed: #guess will be a circle
            n = self.n_starting_points
            radius = half_width if half_width < half_height else half_height
            self.points = [ np.array([
                half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                for x in range( 0, n )
            ]
        else:  #guess will be an horizontal line
            n = self.n_starting_points
            factor = math.floor( half_width / (self.n_starting_points-1) )
            self.points = [ np.array([ math.floor( half_width / 2 ) + x * factor, half_height ])
                for x in range( 0, n )
            ]



    def viz( self ):
        #current state of the snake.
        img = self.image.copy()

        # Drawing lines between points
        point_color = ( 0, 255, 255 )     # BGR RED
        line_color = ( 128, 0, 0 )      # BGR half blue
        thickness = 2                   # Thickness of the lines and circles

        # Draw a line between the current and the next point
        n_points = len( self.points )
        for i in range( 0, n_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        # 0 -> N (Closes the snake)
        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ n_points-1 ] ), line_color, thickness )

        # Drawing circles over points
        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img

    def dist( a, b ):
        #euclidean distance
        return np.sqrt( np.sum( ( a - b ) ** 2 ) )

    def normalize( kernel ):
        #Normalizes a kernel
        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel

    def get_length(self):
        n_points = len(self.points)
        if not self.closed:
            n_points -= 1
        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )



    def uniform( self, p, prev ):
        #The uniform energy.
        avg_dist = self.snake_length / len( self.points )
        un = Snake.dist( prev, p )
        dun = abs( un - avg_dist )

        return dun**2



    def curvture( self, p, prev, next ):
        #The curvture energy
        
        ux = p[0] - prev[0]
        uy = p[1] - prev[1]
        un = math.sqrt( ux**2 + uy**2 )

        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt( vx**2 + vy**2 )

        if un == 0 or vn == 0:
            return 0

        cx = float( vx + ux )  / ( un * vn )
        cy = float( vy + uy ) / ( un * vn )

        cn = cx**2 + cy**2

        return cn

    def rem_orl_pts( self ):
        #Remove overlaping points 

        snake_size = len( self.points )

        for i in range( 0, snake_size ):
            for j in range( snake_size-1, i+1, -1 ):
                if i == j:
                    continue

                curr = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( curr, end )

                if dist < self.MinD:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break

    def add_missing_points( self ):
        #Add points to the spline 
        snake_size = len( self.points )
        for i in range( 0, snake_size ):
            prev = self.points[ ( i + snake_size-1 ) % snake_size ]
            curr = self.points[ i ]
            next = self.points[ (i+1) % snake_size ]
            next2 = self.points[ (i+2) % snake_size ]

            if Snake.dist( curr, next ) > self.MaxD:
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1

    def step( self ):
        #Perform a step in the active contour algorithm
        
        changed = False

        # Computes the length of the snake (used by uniform function)
        self.snake_length = self.get_length()
        new_snake = self.points.copy()

        # Kernels
        sks = ( self.sks, self.sks )
        hks = math.floor( self.sks / 2 )
        e_uniform = np.zeros( sks )
        e_curvture = np.zeros( sks )
        
        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]


            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    p = np.array( [curr[0] + dx, curr[1] + dy] )

                    # Calculates the energy functions on p
                    e_uniform[ dx + hks ][ dy + hks ] = self.uniform( p, prev )
                    e_curvture[ dx + hks ][ dy + hks ] = self.curvture( p, prev, next )
        


            # Normalizes energies
            e_uniform = Snake.normalize( e_uniform )
            e_curvture = Snake.normalize( e_curvture )


            # The sum of all energies for each point

            e_sum = self.alpha * e_uniform \
                    + self.beta * e_curvture \
            # Searches for the point that minimizes the sum of energies e_sum
            emin = np.finfo(np.float64).max
            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy

            # Boundary check
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y

            # Check for changes
            if curr[0] != x or curr[1] != y:
                changed = True

            new_snake[i] = np.array( [ x, y ] )

        self.points = new_snake
        self.rem_orl_pts()
        self.add_missing_points()

        return changed

