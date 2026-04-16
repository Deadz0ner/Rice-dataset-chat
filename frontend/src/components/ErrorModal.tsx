interface ErrorModalProps {
  message: string;
  onClose: () => void;
}

export function ErrorModal({ message, onClose }: ErrorModalProps) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span className="modal-title">Error</span>
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>
        <p className="modal-body">{message}</p>
      </div>
    </div>
  );
}
