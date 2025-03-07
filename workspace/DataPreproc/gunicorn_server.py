
import sys
sys.path.append('/workspace/DataPreproc/')
from ppt_dashapp import server


# To run this using a production server like gunicorn, you need to create a WSGI entry point.

if __name__ == "__main__":
    server.run()

    # Then you can run the application using gunicorn with the following command:
    # gunicorn -w 4 -b 0.0.0.0:8051 gunicorn_server:server