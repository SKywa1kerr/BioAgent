interface ConfirmationDialogProps {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmationDialog({ message, onConfirm, onCancel }: ConfirmationDialogProps) {
  return (
    <div className="confirm-overlay">
      <div className="confirm-dialog">
        <h3>请确认</h3>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="ghost-button" onClick={onCancel}>取消</button>
          <button className="primary-button" onClick={onConfirm}>确认</button>
        </div>
      </div>
    </div>
  );
}
