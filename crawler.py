from typing import Union, List

import io
import bs4
import string
import random
import requests

from collections import deque

import logger


def test():
    crawler = Crawler()
    crawler.crawl(
        'https://en.wikipedia.org/wiki/Hello',
        n_links_per_page=1,
        max_depth=10
    )
    print(len(crawler.data))

    with io.open('data/wiki.txt', 'w', encoding='utf-8') as f:
        f.writelines(crawler.data)

class Crawler:

    def __init__(self):
        self.data = []
        self.visited_pages = set()
        self.item: WebItem = WebItem('wikipedia.org', ['p'])
        self.frontier = deque([])  # A FIFO queue of all pages to explore

    def crawl(
        self, page: Union[str, list],
        n_links_per_page: int=2,
        max_depth: int=5
    ) -> None:

        '''
        Function:
            - Looks for links and raw text (information) to add to the knowledge base.
            - Uses requests library to send GET requests and will parse the document
            that is received.
        Args:
            page:
                - A string or list which represents a document(s) which need to be crawled.
            n_links_per_page:
                - An integer that describes how many links should be accessed per page.
            max_depth:
                - An integer which is the maximum number of links that the web crawler has
                to go to when crawling a page.
        Returns:
            A string with all the necessary text which goes to the corpus.
        '''

        i = 0
        while len(self.visited_pages) < max_depth:
            logger.INFO(f'Visiting - {page}')
            self.visited_pages.add(page)
            # Deque a page from the queue
            page = self.frontier.popleft() if self.frontier else page
            print(self.frontier)

            try: src = requests.get(page).text
            except:
                logger.CRITICAL(f'Request error with page: {page}')
                continue

            soup = bs4.BeautifulSoup(src, 'html.parser')
            for p in soup.find_all('p'):
                text = p.get_text().replace('\n', '')
                if not self._content_is_valid(text): continue
                self.data.append(text)

            n_ = 0  # Number of links found
            for link in self._get_links(soup):
                if not link or link in self.visited_pages: continue
                if not self._url_is_valid(link):
                    logger.CRITICAL(f'Invalid URL: {page}')
                    continue

                if link.startswith('/'):
                    # The link is usually a subdirectory
                    link = 'https://en.wikipedia.org' + link
                    self.frontier.append(link)
                    n_ += 1

                if n_ >= n_links_per_page:
                    break

            i += 1

    def _get_links(self, soup: bs4.BeautifulSoup) -> List[str]:
        return [link.get('href') for link in soup.find_all('a')]

    def _content_is_valid(self, text) -> bool:
        allowed = string.ascii_letters + string.digits + string.punctuation + ' '
        metadata = [1 if char in allowed else 0 for char in text]

        return all(metadata)

    def _url_is_valid(self, url) -> bool:
        blacklisted_exts = ['pdf']
        ext = url.split('.')[-1]

        return ext not in blacklisted_exts


class WebItem:

    '''
    Class:
        - An item that contains relevant information for crawling a site.
        - A WebItem is a site with a name which has properties specific to it.
        (See Args below).
    Args:
        name:
            - A unique identifier for a site to crawl, such as a domain.
        tags:
            - HTML tags which may contains relevant text. E.g: p for paragraphs
            or div with a class
    '''

    def __init__(self, name: str, tags: List[str]):
        self.name = name
        self.tags = tags


if __name__ == '__main__':
    test()