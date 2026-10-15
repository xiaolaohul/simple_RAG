# 三领域中文 RAG 语料

整理日期：2026-10-15。15 份入库文档分为 3 个领域，每个领域 5 份，其中 3 份长文、2 份短文，共 9 长 6 短。

这些文档均有一对一的公开网页来源。NASA、NOAA 英文资料经过中文翻译、概括和段落重组；MDN 中文资料经过节选改写和压缩。它们是基于公开资料的中文整理版本，不是原文全文，也不是原机构发布的官方中文译本。为适配本项目的中文编码器和分块实验，选择了不同篇幅；没有通过重复正文凑字数。

## 目录

- `docs/astronomy/`：天文与太阳系，来源为 NASA。
- `docs/environment/`：海洋与自然环境，来源为 NOAA。
- `docs/computing/`：计算机与 Web 基础，来源为 MDN / Mozilla Contributors。
- `raw/`：15 个源网页的文字快照，已去除脚本、导航等部分页面元素，可能保留少量相关链接；用于溯源，不加入默认索引。
- `sources.json`：标题、原文链接、机构、处理说明、许可、获取日期、字数和文件哈希。
- `corpus_stats.json`：语料及分块统计。
- `source_downloads.json`：源网页下载结果和快照哈希。
- `retrieval_checks.json`：15 条检索自检问题及期望来源；这些问题依据语料设计，不是独立评测集。
- `retrieval_results.json`：本次检索自检的实际来源、分数和结果。

## 字数口径与分块

统计正文时排除首行标题，不把来源链接和元数据算进字数。`正文汉字数`只统计 U+4E00 至 U+9FFF 的汉字；`正文字符数`统计包括英文、数字和标点在内的非空白字符。9 篇长文在这两种口径下均超过 350，其余 6 篇均不超过 350；即使把标题计入全文，长短分组也不变。

程序分块采用字符窗口，不是按汉字数或 token 数切分：每块最多 350 个字符，重叠最多 60 个字符，并优先寻找段落或句子边界。当前 9 篇长文各分成 3 块、6 篇短文各分成 1 块，共 33 块。这里的字数阈值用于安排实验语料，块数仍以实际分块结果为准。

## 文档清单

|编号|领域|中文文档|正文汉字数|正文字符数|块数|原文|
|---|---|---|---:|---:|---:|---|
|AST-01|天文与太阳系|[太阳的结构与能量来源](docs/astronomy/01_sun.md)|568|614|3|[NASA](https://science.nasa.gov/sun/facts/)|
|AST-02|天文与太阳系|[月球的自转、地貌与水冰](docs/astronomy/02_moon.md)|560|604|3|[NASA](https://science.nasa.gov/moon/facts/)|
|AST-03|天文与太阳系|[火星的昼夜、季节与大气](docs/astronomy/03_mars.md)|560|609|3|[NASA](https://science.nasa.gov/mars/facts/)|
|AST-04|天文与太阳系|[木星的基本特征](docs/astronomy/04_jupiter.md)|204|221|1|[NASA](https://science.nasa.gov/jupiter/jupiter-facts/)|
|AST-05|天文与太阳系|[土星与行星环](docs/astronomy/05_saturn.md)|190|206|1|[NASA](https://science.nasa.gov/saturn/facts/)|
|ENV-01|海洋与自然环境|[海洋洋流的驱动与作用](docs/environment/01_ocean_currents.md)|569|616|3|[NOAA](https://www.noaa.gov/education/resource-collections/ocean-coasts/ocean-currents)|
|ENV-02|海洋与自然环境|[水循环与水的储存和迁移](docs/environment/02_water_cycle.md)|581|640|3|[NOAA](https://www.noaa.gov/education/resource-collections/freshwater/water-cycle)|
|ENV-03|海洋与自然环境|[海洋酸化的机制及生态影响](docs/environment/03_ocean_acidification.md)|592|640|3|[NOAA](https://www.noaa.gov/education/resource-collections/ocean-coasts/ocean-acidification)|
|ENV-04|海洋与自然环境|[珊瑚白化是什么](docs/environment/04_coral_bleaching.md)|180|200|1|[NOAA](https://oceanservice.noaa.gov/facts/coral_bleach.html)|
|ENV-05|海洋与自然环境|[潮汐与潮差](docs/environment/05_tides.md)|178|193|1|[NOAA](https://oceanservice.noaa.gov/facts/tides.html)|
|WEB-01|计算机与 Web 基础|[HTTP 的请求、响应与会话](docs/computing/01_http.md)|655|763|3|[MDN](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Guides/Overview)|
|WEB-02|计算机与 Web 基础|[HTML 元素与文档结构](docs/computing/02_html.md)|631|738|3|[MDN](https://developer.mozilla.org/zh-CN/docs/Learn_web_development/Core/Structuring_content/Basic_HTML_syntax)|
|WEB-03|计算机与 Web 基础|[CSS 的规则、样式与应用方式](docs/computing/03_css.md)|647|731|3|[MDN](https://developer.mozilla.org/zh-CN/docs/Learn_web_development/Core/Styling_basics/What_is_CSS)|
|WEB-04|计算机与 Web 基础|[DNS 与域名解析](docs/computing/04_dns.md)|162|204|1|[MDN](https://developer.mozilla.org/zh-CN/docs/Glossary/DNS)|
|WEB-05|计算机与 Web 基础|[JavaScript 的用途](docs/computing/05_javascript.md)|154|249|1|[MDN](https://developer.mozilla.org/zh-CN/docs/Glossary/JavaScript)|

## 运行

在项目根目录运行以下命令，即可建立或刷新正式索引，并使用这 15 份文档问答：

```powershell
.\.venv\Scripts\python.exe rag.py ingest
.\.venv\Scripts\python.exe rag.py chat
```

正式索引保存于 `storage/index.json`。之前的两份演示资料仍在 `examples/docs`，演示索引为 `storage/demo.json`。默认入库目录是 `data/docs`，请不要将 `--docs` 指向整个 `data`，以免把英文快照、说明文件等重复加入知识库。

本次建索引已完成，实际使用 `cuda:0`，得到 33 条 512 维向量。15 条语料自检问题的目标文档均排在首位，详细结果见 `retrieval_results.json`。该结果只说明这组自检通过，不代表独立测试集上的准确率。

可测试：“为什么我们通常看到月球的同一面？”、“海洋酸化为什么影响贝类造壳？”、“HTTP 无状态，网站怎么维持会话？”。

## 来源、署名与许可

NASA 和 NOAA 的文档只整理机构科普文字，不收录图片、标志、视频或第三方媒体。原文链接、获取日期和处理方式均在上表及 `sources.json` 中保留。参考 [NASA 内容使用说明](https://www.nasa.gov/nasa-brand-center/images-and-media/) 和 [NOAA 信息使用问答](https://oceanservice.noaa.gov/about/faq.html)。这些机构没有对本整理版本作出背书。

`docs/computing/` 和对应 `raw/WEB-*.txt` 的来源署名为 **Mozilla Contributors**。原文标题和逐篇链接见上表及 `sources.json`。中文入库版本进行了节选、改写和压缩；按原文的 **CC BY-SA 2.5 或其后续版本**许可提供，参见 [MDN 署名及版权说明](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license) 与 [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/)。分发这些文档时应一并保留对应的署名、来源、修改说明与许可信息。本说明不改变项目程序代码的许可。
