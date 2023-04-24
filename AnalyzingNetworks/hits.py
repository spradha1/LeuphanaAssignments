'''
  Hyperlink-Induced Topic Search (HITS)
'''

# libraries
import numpy as np


# globals
PAGES=10
STEPS=10
SEED=10


# page class
class Page:
    
  def __init__(self) -> None:
    self.auth = 1
    self.hub = 1
    self.incoming_pages = set()
    self.outgoing_pages = set()
   
  def __str__(self) -> str:
    return f'auth: {np.round(self.auth, 3)}; hub: {np.round(self.hub, 3)}; {len(self.incoming_pages)} ins; {len(self.outgoing_pages)} outs'


# main
if __name__ == '__main__':

  np.random.seed(SEED)

  # all pages with both scores equal to 1
  pages = {p:Page() for p in range(PAGES)}

  # randomized links
  links = np.random.randint(low=0, high=PAGES, size=(PAGES,2))
  for u, v in links:
    if u != v:
      pages[u].outgoing_pages.add(v)
      pages[v].incoming_pages.add(u)

  # score updates
  for k in range(STEPS):
    # auth update first
    norm_auth = 0
    for pg in pages.values():
      pg.auth = 0
      for pg_num in pg.incoming_pages:
        pg.auth += pages[pg_num].hub
      norm_auth += np.square(pg.auth)
    norm_auth = np.sqrt(norm_auth)
  
    # hub update
    norm_hub = 0
    for pg in pages.values():
      pg.hub = 0
      for pg_num in pg.outgoing_pages:
        pg.hub += pages[pg_num].auth
      norm_hub += np.square(pg.hub)
    norm_hub = np.sqrt(norm_hub)

    for pg in pages.values():
      pg.auth /= norm_auth
      pg.hub /= norm_hub

  # results
  print(f'Data:\n{links}')
  print(f'After {STEPS} updates:')
  for pg in pages.values():
    print(pg)
