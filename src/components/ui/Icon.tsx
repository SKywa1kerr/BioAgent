import {
  X,
  ChevronRight,
  ChevronLeft,
  ChevronDown,
  ChevronUp,
  Search,
  Settings,
  Moon,
  Sun,
  MessageSquare,
  Send,
  Download,
  HelpCircle,
  Command,
  Globe,
  Trash2,
  Eraser,
  ArrowRight,
  AlertCircle,
  Info,
  CheckCircle2,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelLeft,
  FileText,
  FileSpreadsheet,
  FileDown,
  ZoomIn,
  ZoomOut,
  RefreshCw,
  PlusCircle,
  Languages,
  Keyboard,
  Filter,
  ListFilter,
  GitCompare,
  History,
  Beaker,
  TrendingUp,
} from "lucide-react";

const ICONS = {
  close: X,
  "chevron-right": ChevronRight,
  "chevron-left": ChevronLeft,
  "chevron-down": ChevronDown,
  "chevron-up": ChevronUp,
  search: Search,
  settings: Settings,
  moon: Moon,
  sun: Sun,
  message: MessageSquare,
  send: Send,
  download: Download,
  help: HelpCircle,
  command: Command,
  globe: Globe,
  trash: Trash2,
  eraser: Eraser,
  "arrow-right": ArrowRight,
  alert: AlertCircle,
  info: Info,
  success: CheckCircle2,
  loader: Loader2,
  "panel-close": PanelLeftClose,
  "panel-open": PanelLeftOpen,
  "panel-toggle": PanelLeft,
  file: FileText,
  spreadsheet: FileSpreadsheet,
  "file-download": FileDown,
  "zoom-in": ZoomIn,
  "zoom-out": ZoomOut,
  refresh: RefreshCw,
  add: PlusCircle,
  language: Languages,
  keyboard: Keyboard,
  filter: Filter,
  "list-filter": ListFilter,
  compare: GitCompare,
  history: History,
  lab: Beaker,
  trend: TrendingUp,
} as const;

export type IconName = keyof typeof ICONS;

interface IconProps {
  name: IconName;
  size?: number;
  strokeWidth?: number;
  className?: string;
  "aria-hidden"?: boolean | "true" | "false";
  "aria-label"?: string;
}

export function Icon({ name, size = 16, strokeWidth = 1.75, className, ...rest }: IconProps) {
  const Component = ICONS[name];
  return (
    <Component
      size={size}
      strokeWidth={strokeWidth}
      className={className}
      aria-hidden={rest["aria-label"] ? rest["aria-hidden"] ?? false : true}
      aria-label={rest["aria-label"]}
    />
  );
}
