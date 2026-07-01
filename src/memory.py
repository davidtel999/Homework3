import json
from pathlib import Path
from langchain_openai import ChatOpenAI

# Connect paths to the project root to guarantee consistent file resolution, 
# regardless of the current working directory at runtime:
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_PATH = BASE_DIR / "chat_memory_profile.json"

class ShortTermMemory:
    def __init__(self, max_turns: int = 6):
        # Manages the active conversational memory context.
        self.max_turns = max_turns
        
        # System messages are intentionally never stored here - they are stored
        # in main.py's `base_instruction` and get added directly to the message
        # list sent to the LLM each turn, and not into short-term memory. 
        self.messages = []
        
        # `self.messages` gets trimmed to the last max_turns exchanges to keep the
        # prompts small, but we keep a full session log for long-term memory summary:
        self.full_session_log = []

    def add_user_message(self, content: str):
        entry = {"role": "user", "content": content}
        self.messages.append(entry)
        self.full_session_log.append(entry)
        self._trim()

    def add_assistant_message(self, content: str):
        entry = {"role": "assistant", "content": content}
        self.messages.append(entry)
        self.full_session_log.append(entry)
        self._trim()

    def _trim(self):
        # Keep only the last N turns (user + assistant pairs).
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_recent_history(self) -> list[dict]:
        return self.messages

class LongTermMemoryManager:
    def __init__(self, filepath: str = None):
        # If no path is given, always use the absolute default -
        # so you always get the same file, no matter which directory you ran the script from.
        self.filepath = Path(filepath).resolve() if filepath else DEFAULT_MEMORY_PATH
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def load_summary(self) -> str:
        # Loads the persistent long-term profile if it exists. Returns '' if not.
        if not self.filepath.exists():
            return ""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("summary", "")
        except Exception:
            return ""

    def save_summary(self, short_term_history: list[dict]):
        # Uses an LLM call to extract and compress long-term traits, then saves to disk.
        if not short_term_history:
            return

        compression_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a memory consolidation engine. Analyze the provided chat history "
                    "and extract a consolidated summary containing exactly three things:\n"
                    "1. Topics discussed\n"
                    "2. User preferences expressed\n"
                    "3. Unresolved questions\n"
                    "Keep it concise and factual. Do not include raw conversational pleasantries."
                ),
            },
            {
                "role": "user",
                "content": f"Chat History:\n{json.dumps(short_term_history, indent=2)}",
            },
        ]

        try:
            response = self.llm.invoke(compression_prompt)
            summary_text = response.content

            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"summary": summary_text}, f, indent=4, ensure_ascii=False)
            print(f"Long-term memory saved to: {self.filepath}")
        except Exception as e:
            print(f"Failed to save long-term memory: {e}")

    def clear(self):
        # Deletes the long-term memory file, if it exists.
        # Using it guarantees the deletion of the exact file this class actually reads from.
        if self.filepath.exists():
            self.filepath.unlink()
            print(f"Deleted long-term memory file: {self.filepath}")
        else:
            print(f"No long-term memory file found at: {self.filepath}")