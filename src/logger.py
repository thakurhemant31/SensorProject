import logging  #the login message amint that can be track
import os  #provide a reading and writting to file system
from  datetime import datetime


LOG_FILE=f"{datetime.now().strftime("%m_%d_%y_ %H_%M_%s")}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]%(lineno)d %(name)s -%(levelname)s- %(message)s",
    level=logging.INFO
)