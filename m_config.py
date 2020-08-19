# import xml.etree.ElementTree as ET

# root = ET.Element("parameters")

# title = ET.SubElement(root, "app_title")
# title.text = "M Face Recognition"

# models = ET.SubElement(root, "model_paths")

# FD_model = ET.SubElement(models, "face_detection_model_paths")
# FD_model.text = "models/retinaface_mobinet/mobilenet_0_25_Final.pth"

# FR_model = ET.SubElement(models, "face_recognition_model_paths")
# FR_model.text = "models/insightface-r100-ii/model"

# database = ET.SubElement(root, "database")
# database.text = "m_database"

# server = ET.SubElement(root, "server")
# HOST = ET.SubElement(server, "HOST")
# HOST.text = "localhost"
# PORT = ET.SubElement(server, "PORT")
# PORT.text = "8022"

# arguments = ET.SubElement(root, "arguments")

# RESET_FLAG_FACE_DETECT = ET.SubElement(arguments, "RESET_FLAG_FACE_DETECT")
# RESET_FLAG_FACE_DETECT.text = str(3)

# PADDING_FACE = ET.SubElement(arguments, "PADDING_FACE")
# PADDING_FACE.text = str(0.1)

# HEIGHT_FRAME = ET.SubElement(arguments, "HEIGHT_FRAME")
# HEIGHT_FRAME.text = str(500)

# IOU_THRESHOLD = ET.SubElement(arguments, "IOU_THRESHOLD")
# IOU_THRESHOLD.text = str(0.5)

# RECOGNIZE_THRESHOLD = ET.SubElement(arguments, "RECOGNIZE_THRESHOLD")
# RECOGNIZE_THRESHOLD.text = str(1.24)

# mydata = ET.tostring(root)
# myfile = open("config.xml", "wb")
# myfile.write(mydata)
# myfile.close()


import xml.etree.ElementTree as ET
tree = ET.parse("config.xml")
root = tree.getroot()

TITLE = root.find("app_title").text

model_paths = root.find("model_paths")
MODEL_FACE_DETECTION_PATH = model_paths.find("face_detection_model_paths").text
MODEL_FACE_RECOGNIZE_PATH = model_paths.find("face_recognition_model_paths").text

DATABASE = root.find("database").text

server = root.find("server")
HOST = server.find("HOST").text
PORT = server.find("PORT").text

arguments = root.find("arguments")
RESET_FLAG_FACE_DETECT = int(arguments.find("RESET_FLAG_FACE_DETECT").text)
PADDING_FACE = float(arguments.find("PADDING_FACE").text)
HEIGHT_FRAME = int(arguments.find("HEIGHT_FRAME").text)
IOU_THRESHOLD = float(arguments.find("IOU_THRESHOLD").text)
RECOGNIZE_THRESHOLD = float(arguments.find("RECOGNIZE_THRESHOLD").text)

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    pass
