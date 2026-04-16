export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  sources?: ChatSource[];
  note?: string | null;
}

export interface ChatSource {
  row_id: string;
  preview: string;
  score?: number | null;
  metadata: Record<string, string | number | null>;
}

export interface ChatRequest {
  message: string;
  history: { role: string; content: string }[];
}

export interface ChatResponse {
  answer: string;
  grounded: boolean;
  sources: ChatSource[];
  note?: string | null;
}
