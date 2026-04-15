import { apiPost } from "./api";
import type { ChatRequest, ChatResponse } from "../types/chat";

export function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  return apiPost<ChatResponse, ChatRequest>("/chat", payload);
}
