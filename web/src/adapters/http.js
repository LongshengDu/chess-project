export async function request(path, data, method = 'POST', signal) {
  const response = await fetch(`/api/platform${path}`, {
    method: data === undefined ? 'GET' : method, signal,
    ...(data === undefined ? {} : {headers: {'Content-Type':'application/json'}, body: JSON.stringify(data)}),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error || `Local request failed (${response.status})`);
  }
  return response.json();
}
