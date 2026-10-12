import unittest

from langchain_core.messages import HumanMessage, AIMessage

from agentic_chatbot.chatbot import collect_stream_text, extract_message_text


class ChatbotStreamHelpersTest(unittest.TestCase):
    def test_extract_message_text_from_string_and_blocks(self):
        self.assertEqual(extract_message_text(HumanMessage(content="Hi there")), "Hi there")
        self.assertEqual(
            extract_message_text(
                AIMessage(content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}])
            ),
            "Hello world",
        )

    def test_collect_stream_text_ignores_empty_chunks(self):
        class EmptyMessage:
            content = None

        stream = [
            (HumanMessage(content="Hi"), {}),
            (AIMessage(content=" there"), {}),
            (EmptyMessage(), {}),
        ]

        self.assertEqual(collect_stream_text(stream), "Hi there")


if __name__ == "__main__":
    unittest.main()
