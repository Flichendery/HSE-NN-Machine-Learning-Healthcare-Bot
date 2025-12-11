import os
import telebot
import time
from dotenv import load_dotenv
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton, ReplyKeyboardMarkup, InputFile

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

def create_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
    keyboard.add(KeyboardButton("Авторы"))
    return keyboard


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    keyboard = create_keyboard()
    user_id = message.chat.id
    response = ("🌠 Привет!\n\nЭтот бот помогает заботиться о здоровье.\n"
                "Расскажи ему о своём текущем состоянии и он даст рекомендации как лучше поступить.\n\n"
                "Но помни! Если появились **серьёзные** проблемы, то нужно сразу обращаться ко врачу!"
    )
    
    bot.send_message(user_id, response, parse_mode="Markdown", reply_markup=create_keyboard())


@bot.message_handler(func=lambda message: message.text == "Авторы")
def show_authors(message):
    user_id = message.chat.id
    authors_info = (
        "🧑‍💻 Об авторах проекта\n\n"
        "Бот сделан Артёмом Шеховцовым, Дмитрием Лепа и Владимиром Заворохиным\n\n"
        "Спасибо за использование!"
    )

    bot.send_message(user_id, authors_info, parse_mode="Markdown")


# Получение сообщений пользователя и их обработка
@bot.message_handler(content_types=['text'])
def get_message(message):
    user_id = message.chat.id
    text = message.text
    
    reversed_text = text[::-1] 
    response = f"{reversed_text}"
    
    bot.send_message(user_id, response, parse_mode="Markdown")



# Запуск
if __name__ == '__main__':
    while True:
        try:
            print("Bot started!")
            print(os.getcwd())
            bot.polling(none_stop=True, interval=0, timeout=60, long_polling_timeout=60)
        except Exception as e:
            print(f"Error: {e}")
            print("Reload after 10 seconds")
            time.sleep(10)