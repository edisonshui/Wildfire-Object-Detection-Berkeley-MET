from ultralytics import YOLO
import cv2

model = YOLO("hope-to-god.pt")
src = cv2.VideoCapture("Inside a Wildfire - Dramatic Drone Footage! Ep. 193a..mp4")

w = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = src.get(cv2.CAP_PROP_FPS) or 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (w, h))

while True:
    ok, frame = src.read()
    if not ok:
        break
    res = model(frame, verbose = False)
    plotted = res[0].plot()
    cv2.imshow("YOLO", plotted)
    out.write(plotted)
    if cv2.waitKey(1) & 0xFF == 27:
        break

src.release()
out.release()
cv2.destroyAllWindows()

