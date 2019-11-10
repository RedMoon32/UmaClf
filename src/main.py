import os
from constants import result_clf, greeting, token
from ml_model import FootballersModel,logger
import matplotlib.pyplot as plt
from telegram.ext import Updater, CommandHandler
from telegram.ext import MessageHandler, Filters

f = FootballersModel()

updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher


def start(update, context):
    """ Greeting message"""
    logger.info("New /start command")
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=greeting)


def classify(update, context):
    """ Function which receives image and tries to classify it"""
    logger.info("New Classification request")
    fid = update.message.photo[0].file_id
    file = context.bot.getFile(fid)
    file.download(str(fid))
    p = plt.imread(str(fid))
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=result_clf.format(f.predict(p)))
    os.remove(str(fid))


start_handler = CommandHandler('start', start)
image_handler = MessageHandler(Filters.photo, classify)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(image_handler)

if __name__ == "__main__":
    updater.start_polling()
