import math

# MediaPipe Face Mesh landmark indices for the eye contours.
# These are used as six points per eye for EAR:
# p1/p4 are horizontal endpoints; p2/p3 and p5/p6 form vertical distances.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calculate_ear(points):
    """Calculate Eye Aspect Ratio from six eye landmark points."""
    if len(points) != 6:
        raise ValueError("EAR requires exactly 6 points.")

    p1, p2, p3, p4, p5, p6 = points

    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)
