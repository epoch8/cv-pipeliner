import subprocess

while True:
    label_studio_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port=8555", "--server.enableXsrfProtection=false"],
        stdout=subprocess.PIPE
    )
    output = '\n'.join([x.decode() for x in label_studio_process.communicate() if x])
