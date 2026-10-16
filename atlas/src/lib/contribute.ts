// Links into the repository's issue forms, pre-filled where GitHub allows (field ids match the forms
// in .github/ISSUE_TEMPLATE/).
export const REPO_URL = 'https://github.com/Iammohithhh/MLIPs_at';

export function correctionUrl(collection: string, id: string, statement?: string): string {
  const p = new URLSearchParams({
    template: 'correction.yml',
    title: `[correction] ${collection}/${id}`,
    record: `${collection}/${id}`,
  });
  if (statement) p.set('problem', `> ${statement.slice(0, 600)}\n\n`);
  return `${REPO_URL}/issues/new?${p.toString()}`;
}

export const newSourceUrl = () => `${REPO_URL}/issues/new?template=new-source.yml`;
export const failureCaseUrl = () => `${REPO_URL}/issues/new?template=failure-case.yml`;
