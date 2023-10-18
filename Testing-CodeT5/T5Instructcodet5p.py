from transformers import AutoModel, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m-bimodal"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

code = """def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG resources.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image"""

input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=20)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# Convert a string of SVG resources to an image.