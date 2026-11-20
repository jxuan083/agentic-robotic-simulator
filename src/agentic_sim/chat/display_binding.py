"""Which question a line typed in a GUI answers: the one the window *displayed* when the line was started.

The window learns about a new question on a background thread (reports, task state) and draws it later on the UI
thread. A line is bound to the question on screen when its input box went from empty to non-empty: keystrokes,
input-method commits and pastes all change the box content, so all of them start a line the same way. Clearing the box
releases the binding; the next content binds again to whatever is displayed then. The agent then checks the bound id
against the question that is open (ChatAgent.handle(text, shown=...)), so a question the user has not seen yet never
receives the answer.

  b = DisplayBinding()
  b.displayed(qid)        # UI thread, when it draws a question (report line, question bar) or shows none (None)
  b.text_changed(text)    # UI thread, on every change of the input box content
  shown = b.take()        # on send: the bound id (released)
"""

from __future__ import annotations

_UNSET = object()


class DisplayBinding:
    def __init__(self):
        self.on_screen = None          # question id the window shows now (None: no question)
        self._bound = _UNSET           # id displayed when the current line was started

    def displayed(self, qid):
        self.on_screen = qid

    def text_changed(self, text: str):
        if not text.strip():
            self._bound = _UNSET
        elif self._bound is _UNSET:
            self._bound = self.on_screen

    @property
    def bound(self):
        return self.on_screen if self._bound is _UNSET else self._bound

    def take(self):
        shown = self.bound
        self._bound = _UNSET
        return shown
