import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackBody?: string;
  retryLabel?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-card">
          <h3>{this.props.fallbackTitle || "Rendering Error"}</h3>
          <p>{this.props.fallbackBody || "An unexpected error occurred."}</p>
          <p className="error-boundary-detail">{this.state.error?.message}</p>
          <button className="primary-button" onClick={this.handleRetry}>
            {this.props.retryLabel || "Retry"}
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
