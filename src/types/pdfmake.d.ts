declare module "pdfmake/build/pdfmake" {
  const pdfMake: {
    createPdf: (def: unknown) => {
      getBase64: (cb: (data: string) => void) => void;
      download: (filename?: string) => void;
    };
    vfs?: Record<string, string>;
    fonts?: Record<string, { normal: string; bold: string; italics: string; bolditalics: string }>;
  };
  export default pdfMake;
}
