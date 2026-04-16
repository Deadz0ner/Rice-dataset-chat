import { useEffect, useState } from "react";

import { ChatInput } from "../components/ChatInput";
import { LoadingDots } from "../components/LoadingDots";
import { MessageBubble } from "../components/MessageBubble";
import { UploadPanel } from "../components/UploadPanel";
import { useAutoScroll } from "../hooks/useAutoScroll";
import { sendChatMessage } from "../services/chatService";
import { fetchDatasetSummary, uploadDataset } from "../services/datasetService";
import type { ChatMessage } from "../types/chat";
import type { DatasetSummary } from "../types/dataset";

const starterMessage: ChatMessage = {
  id: "assistant-welcome",
  role: "assistant",
  content:
    "Upload your Excel dataset, then ask questions in natural language. All answers are grounded strictly in the uploaded data.",
};

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([starterMessage]);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useAutoScroll(messages);

  useEffect(() => {
    async function loadSummary() {
      try {
        const summary = await fetchDatasetSummary();
        setDatasetSummary(summary);
      } catch {
        setDatasetSummary(null);
      }
    }

    void loadSummary();
  }, []);

  async function handleUpload(file: File) {
    setError(null);
    setIsUploading(true);
    try {
      const response = await uploadDataset(file);
      setDatasetSummary(response.summary);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Dataset loaded: ${response.summary.file_name}. You can start asking questions now.`,
        },
      ]);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Dataset upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSend(message: string) {
    setError(null);
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: message,
    };

    setMessages((current) => [...current, userMessage]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage({ message });
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        note: response.note,
        sources: response.sources,
      };
      setMessages((current) => [...current, assistantMessage]);
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : "Unable to fetch response.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <UploadPanel datasetSummary={datasetSummary} uploading={isUploading} onUpload={handleUpload} />

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Grounded chat</p>
            <h1>Rice Dataset Assistant</h1>
          </div>
          <p className="chat-subtitle">
            Ask questions in plain English about the uploaded dataset.
          </p>
        </header>

        <div className="chat-window" ref={scrollRef}>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading ? (
            <div className="message message-assistant">
              <div className="message-header">
                <span>Rice Analyst</span>
              </div>
              <LoadingDots />
            </div>
          ) : null}
        </div>

        {error ? <div className="error-banner">{error}</div> : null}
        <ChatInput disabled={isLoading} onSubmit={handleSend} />
      </section>
    </main>
  );
}
