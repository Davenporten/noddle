use dashmap::DashMap;
use std::sync::Arc;

/// A single turn in a conversation.
#[derive(Debug, Clone)]
pub struct Turn {
    pub prompt:   String,
    pub response: String,
}

/// Conversation history for a single session.
#[derive(Debug, Default)]
pub struct Session {
    pub turns: Vec<Turn>,
}

impl Session {
    /// Build the full conversation context to prepend to the next prompt.
    /// Format mirrors what most instruction-tuned models expect.
    pub fn context_for_next_prompt(&self, next_prompt: &str) -> String {
        let mut ctx = String::new();
        for turn in &self.turns {
            ctx.push_str("[USER]: ");
            ctx.push_str(&turn.prompt);
            ctx.push_str("\n[ASSISTANT]: ");
            ctx.push_str(&turn.response);
            ctx.push('\n');
        }
        ctx.push_str("[USER]: ");
        ctx.push_str(next_prompt);
        ctx
    }

    pub fn add_turn(&mut self, prompt: String, response: String) {
        self.turns.push(Turn { prompt, response });
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }
}

/// Thread-safe store of all active sessions, keyed by session ID.
#[derive(Clone, Default)]
pub struct SessionStore {
    sessions: Arc<DashMap<String, Session>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a session for the given ID.
    pub fn get_or_create(&self, session_id: &str) -> dashmap::mapref::one::RefMut<'_, String, Session> {
        if !self.sessions.contains_key(session_id) {
            self.sessions.insert(session_id.to_string(), Session::default());
        }
        self.sessions.get_mut(session_id).unwrap()
    }

    /// Record a completed turn in a session.
    pub fn record_turn(&self, session_id: &str, prompt: String, response: String) {
        self.get_or_create(session_id).add_turn(prompt, response);
    }

    pub fn remove(&self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_for_first_prompt_has_no_history() {
        let session = Session::default();
        let ctx = session.context_for_next_prompt("hello");
        assert_eq!(ctx, "[USER]: hello");
    }

    #[test]
    fn context_includes_previous_turns() {
        let mut session = Session::default();
        session.add_turn("hello".into(), "hi there".into());
        let ctx = session.context_for_next_prompt("how are you?");
        assert!(ctx.contains("[USER]: hello"));
        assert!(ctx.contains("[ASSISTANT]: hi there"));
        assert!(ctx.ends_with("[USER]: how are you?"));
    }

    #[test]
    fn session_store_get_or_create() {
        let store = SessionStore::new();
        let _ = store.get_or_create("abc");
        assert_eq!(store.active_count(), 1);
        let _ = store.get_or_create("abc");
        assert_eq!(store.active_count(), 1); // same session, not duplicated
    }

    #[test]
    fn session_store_remove() {
        let store = SessionStore::new();
        store.record_turn("abc", "hi".into(), "hello".into());
        assert_eq!(store.active_count(), 1);
        store.remove("abc");
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn turn_count_increments() {
        let mut session = Session::default();
        assert_eq!(session.turn_count(), 0);
        session.add_turn("a".into(), "b".into());
        session.add_turn("c".into(), "d".into());
        assert_eq!(session.turn_count(), 2);
    }
}
