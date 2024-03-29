class shape_packet:

    def __init__(self):
        self.horizontal_threshold = 20
        self.packet_type = 0
        self.bbox = (0,0,0,0)
        self.blob_img = []
        self.centroid = []
        
    def set_centroid(self):
        total_rows, total_cols = self.blob_img.shape

        # Calculate the sum of x and y coordinates
        sum_x = 0
        sum_y = 0
        total_points = 0
    
        for row in range(total_rows):
            for col in range(total_cols):
                if self.blob_img[row][col]:
                    sum_y += col
                    sum_x += row
                    total_points += 1
    
        # Avoid division by zero
        if total_points == 0:
            return None, None
        
        self.centroid = (int(sum_x / total_points + self.bbox[0]), int(sum_y / total_points+self.bbox[1]))

    def __eq__(self, other):
        return self.centroid == other.centroid

    def __lt__(self, other):
        if abs(other.centroid[0] - self.centroid[0]) <= self.horizontal_threshold:
            if self.centroid[1] - other.centroid[1] < 0 :
                return True
            else :
                return False
        elif other.centroid[0] - self.centroid[0] > self.horizontal_threshold:
            return True
        elif self.centroid[0] - other.centroid[0] > self.horizontal_threshold:
            return False


class Window_partial_cells:
    def __init__(self, window_coordinates):
        self.type_4_pkts = []
        self.type_8_pkts = []
        self.window_coordinates = window_coordinates