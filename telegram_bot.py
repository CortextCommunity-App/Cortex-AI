import os
import io
import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

BASE_API = "http://127.0.0.1:8000/api/v1/ai"
BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8698705561:AAF6HBevJbmoTqOWYe7bn_RiVyt57Kn0KXc"
)
ALLOWED_TOPIC_ID = 2945


async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat:
        return

    chat_type = update.effective_chat.type
    thread_id = update.message.message_thread_id

    if update.message.from_user and update.message.from_user.is_bot:
        return

    if chat_type in ["group", "supergroup"]:
        if thread_id != ALLOWED_TOPIC_ID:
            return

    user_query = (update.message.caption or update.message.text or "").strip()

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        message_thread_id=thread_id,
        action="typing",
    )

    session_key = (
        f"tg_{update.effective_chat.id}_{thread_id}"
        if thread_id
        else f"tg_{update.effective_chat.id}"
    )
    username = (
        update.effective_user.first_name
        if update.effective_user and update.effective_user.first_name
        else "Student"
    )

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            is_image_doc = bool(
                update.message.document
                and update.message.document.mime_type
                and update.message.document.mime_type.startswith("image/")
            )

            if update.message.photo:
                photo_file = await update.message.photo[-1].get_file()
                img_bytes = await photo_file.download_as_bytearray()
                files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
                data = {"prompt": user_query if user_query else "What is in this image?"}
                res = await client.post(f"{BASE_API}/analyze-image", data=data, files=files)
                res_data = res.json()
                
                print("Backend Image Response:", res_data)

                if isinstance(res_data, dict):
                    answer = (
                        res_data.get("response")
                        or res_data.get("result")
                        or res_data.get("description")
                        or res_data.get("analysis")
                        or res_data.get("message")
                        or res_data.get("answer")
                        or str(res_data)
                    )
                else:
                    answer = str(res_data)

            elif is_image_doc and update.message.document:
                doc_img_file = await update.message.document.get_file()
                img_bytes = await doc_img_file.download_as_bytearray()
                files = {"file": (update.message.document.file_name or "image.jpg", io.BytesIO(img_bytes), update.message.document.mime_type or "image/jpeg")}
                data = {"prompt": user_query if user_query else "What is in this image?"}
                res = await client.post(f"{BASE_API}/analyze-image", data=data, files=files)
                res_data = res.json()
                answer = res_data.get("analysis", res_data.get("answer", "Analysis complete."))

            elif (
                update.message.document
                and update.message.document.mime_type == "application/pdf"
            ):
                doc_file = await update.message.document.get_file()
                pdf_bytes = await doc_file.download_as_bytearray()
                files = {
                    "file": (
                        update.message.document.file_name or "document.pdf",
                        io.BytesIO(pdf_bytes),
                        "application/pdf",
                    )
                }
                res = await client.post(f"{BASE_API}/upload-doc", files=files)
                res_data = res.json()
                answer = res_data.get(
                    "message",
                    "Document uploaded and indexed successfully into ChromaDB!",
                )

            else:
                if not user_query:
                    return
                payload = {
                    "query": user_query,
                    "session_id": session_key,
                    "username": username,
                }
                res = await client.post(f"{BASE_API}/agent-ask", json=payload)
                res_data = res.json()
                answer = res_data.get("answer", "No response received.")

            try:
                await update.message.reply_text(
                    answer,
                    message_thread_id=thread_id,
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                await update.message.reply_text(
                    answer, message_thread_id=thread_id
                )

    except Exception as e:
        await update.message.reply_text(
            f"Error reaching Cortex backend: {e}", message_thread_id=thread_id
        )


if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("ask", chat_handler))
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.PHOTO | filters.Document.ALL)
            & ~filters.COMMAND,
            chat_handler,
        )
    )

    print("Cortex Telegram Bot is running with Vision and Document support...")
    app.run_polling()