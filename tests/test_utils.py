import base64
from io import BytesIO
from PIL import Image
import pytest
import PIL.Image


def test_base64_to_image(
        base64_string: str = "iVBORw0KGgoAAAANSUhEUgAAAEIAAAARCAYAAABpTnqxAAAAAXNSR0IArs4c6QAAAWFJREFUWEftl9HRRDAQgNeTMZ5oISVQAiVogRKihCiBEiiBEighLfBoPOWfzZ0bRwz3v5zL8GRMPmO/7K6sMU2TgPsC4xbxyIJbxLMapAjTNI11dTDGRJqmr+d7mbNk18yZirsCg7EpRXDOBSEE+r4H13WljCMRKuZIxFWYXRHPXYI8zyFJklMiVMyRiKswuyI8zxNd10EURVBV1aZsVAH+MiNF9H2PvQCKojjaQIjjGBhjcp1OjG3bj78GNrw8z6WQYRg2QhzHkQLmMpkX6MJsSqNtW+H7/psIlFDXNfi+rywRHZhTItAK5xwIIadF/BqzEUEpFVmWged5QCkFvMemiWWxPFMsU0YHZiMCO38QBChA7j42UpTRNA0KUWaEDsybCAwadz8Mw03AZVlKQfPhas6IJTOO42t4syzLOMOsu/K3mN1zxOF/dLXgm4Obajz4z/fvzhqfvEwbEZ8ErevaP+CpW7pfhQ58AAAAAElFTkSuQmCC"
):
    byte_stream = base64.b64decode(base64_string)

    image = Image.open(BytesIO(byte_stream))
    PIL.Image.Image.show(image)


str1 = "iVBORw0KGgoAAAANSUhEUgAAAEIAAAARCAYAAABpTnqxAAAAAXNSR0IArs4c6QAAAWFJREFUWEftl9HRRDAQgNeTMZ5oISVQAiVogRKihCiBEiiBEighLfBoPOWfzZ0bRwz3v5zL8GRMPmO/7K6sMU2TgPsC4xbxyIJbxLMapAjTNI11dTDGRJqmr+d7mbNk18yZirsCg7EpRXDOBSEE+r4H13WljCMRKuZIxFWYXRHPXYI8zyFJklMiVMyRiKswuyI8zxNd10EURVBV1aZsVAH+MiNF9H2PvQCKojjaQIjjGBhjcp1OjG3bj78GNrw8z6WQYRg2QhzHkQLmMpkX6MJsSqNtW+H7/psIlFDXNfi+rywRHZhTItAK5xwIIadF/BqzEUEpFVmWged5QCkFvMemiWWxPFMsU0YHZiMCO38QBChA7j42UpTRNA0KUWaEDsybCAwadz8Mw03AZVlKQfPhas6IJTOO42t4syzLOMOsu/K3mN1zxOF/dLXgm4Obajz4z/fvzhqfvEwbEZ8ErevaP+CpW7pfhQ58AAAAAElFTkSuQmCC"
str2 = "R0lGODlhQgARAPcAAP//////zP//mf//Zv//M///AP/M///MzP/Mmf/MZv/MM//MAP+Z//+ZzP+Zmf+ZZv+ZM/+ZAP9m//9mzP9mmf9mZv9mM/9mAP8z//8zzP8zmf8zZv8zM/8zAP8A//8AzP8Amf8AZv8AM/8AAMz//8z/zMz/mcz/Zsz/M8z/AMzM/8zMzMzMmczMZszMM8zMAMyZ/8yZzMyZmcyZZsyZM8yZAMxm/8xmzMxmmcxmZsxmM8xmAMwz/8wzzMwzmcwzZswzM8wzAMwA/8wAzMwAmcwAZswAM8wAAJn//5n/zJn/mZn/Zpn/M5n/AJnM/5nMzJnMmZnMZpnMM5nMAJmZ/5mZzJmZmZmZZpmZM5mZAJlm/5lmzJlmmZlmZplmM5lmAJkz/5kzzJkzmZkzZpkzM5kzAJkA/5kAzJkAmZkAZpkAM5kAAGb//2b/zGb/mWb/Zmb/M2b/AGbM/2bMzGbMmWbMZmbMM2bMAGaZ/2aZzGaZmWaZZmaZM2aZAGZm/2ZmzGZmmWZmZmZmM2ZmAGYz/2YzzGYzmWYzZmYzM2YzAGYA/2YAzGYAmWYAZmYAM2YAADP//zP/zDP/mTP/ZjP/MzP/ADPM/zPMzDPMmTPMZjPMMzPMADOZ/zOZzDOZmTOZZjOZMzOZADNm/zNmzDNmmTNmZjNmMzNmADMz/zMzzDMzmTMzZjMzMzMzADMA/zMAzDMAmTMAZjMAMzMAAAD//wD/zAD/mQD/ZgD/MwD/AADM/wDMzADMmQDMZgDMMwDMAACZ/wCZzACZmQCZZgCZMwCZAABm/wBmzABmmQBmZgBmMwBmAAAz/wAzzAAzmQAzZgAzMwAzAAAA/wAAzAAAmQAAZgAAMwAAAPn5+ff39+/v7+fn59/f39fX18/Pz8fHx8DAwLi4uLCwsKCgoP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAEAAOQALAAAAABCABEAAAj/ALEJHEiwoMGDCBMqXMiwocOHECNKnJiQnEWL3y6SM4jxIseOGkOCFKkxY0iBGrWNy+aRoEWVLDcWfLmS5EWYNmnGbFlym7ec5L75tPhR6M+cRoEm1YjyYkaTNp8qDToVKkmpTLOFE/dNKDluXcWFy6aVq1ew38SS3dp129ewY8u2fZs2LtuzcLOh7NYtpLejFvn6BUxOsMa/Gg1fRHxRsUXGTbdB9RpSslO3lSdj1mgZ4+aLnYNujty3rzdtIreVJnc69erWlV+jjl2Y9WyZGLOB6/ZNm9Xcu3v/Dqqbt2+R34oLR678OFFsFsOZzJZWpHSL1MVZn1495HVy2bdjJu+OOxu3kN12gj+fWL159O7ZN44P/znQ+/jz62dKsb9/6P8FSFFAADs="