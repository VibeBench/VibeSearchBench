"""Prompt templates for query synthesis (final_query, initial_query, user_persona).

Each prompt has a ZH and EN variant. The caller picks the right one based on the
task's ``language`` field so that the synthesised output matches the source language.
"""

# ===========================================================================
# Chinese prompts
# ===========================================================================

FINAL_QUERY_PROMPT_ZH = """\
你是一名信息需求分析专家。你的任务是将一组分阶段的用户子查询（user_queries）整合为一个**语义完整、描述清晰的综合查询（final_query）**。

## 输入

### 用户子查询（按时间顺序）
{user_queries_text}

### 参考知识图谱（Ground Truth）
以下是该查询的期望输出结构，以知识图谱三元组形式呈现。请参考这些信息来理解用户真正需要的信息粒度和输出格式：
{graph_text}

## 要求

1. **语义完整性**：final_query 必须涵盖所有子查询中的每一个信息需求，不得遗漏任何约束条件（如时间范围、数量限制、筛选条件、排序要求等）。
2. **不改变语义**：不得简化、弱化或改变原始查询中的任何约束。例如"前20"不能改为"推荐一些"，"2018年以前"不能改为"比较早的"。
3. **输出格式明确化**：参考知识图谱的结构，在 final_query 中明确期望的输出格式和信息粒度。例如：
   - 如果图谱中包含"歌手→举办→演唱会→包含场次→具体日期"的层次，query 中应明确要求列出具体场次信息。
   - 如果图谱中包含筛选后的子集，query 中应明确筛选条件和期望结果。
4. **连贯自然**：final_query 应该是一段连贯的自然语言文本，而不是简单的子查询拼接。可以使用分段或分点的方式组织，但整体应读起来像一个用户一次性提出的完整需求描述。
5. **保持用户视角**：使用第一人称，保持原始查询中的口语化表达风格。

## 输出

请直接输出 final_query 的文本内容，不要输出其他任何解释或前缀。"""

INITIAL_QUERY_PROMPT_ZH = """\
你是一名用户行为模拟专家。你的任务是根据一个完整的信息需求（final_query），生成一个**模糊、笼统的初始查询（initial_query）**，模拟真实用户在第一次向AI助手提问时的表达方式。

## 输入

### 完整信息需求（final_query）
{final_query}

### 原始用户子查询（参考）
{user_queries_text}

## 要求

1. **保留核心主题**：initial_query 必须包含用户的核心关注领域/话题，让 AI 助手知道大致方向。
2. **刻意模糊化**：去掉所有具体的约束条件，包括但不限于：
   - 具体的数量要求（如"前20"、"不低于24小时"）
   - 具体的时间范围（如"2025年"、"2018年以前"）
   - 具体的平台/渠道限制（如"淘宝和抖音"）
   - 具体的属性筛选（如"油皮"、"哑光"、"预算500元"）
   - 具体的输出格式要求
3. **模拟真实用户**：initial_query 应该像一个真实用户随口提出的问题，简短、自然、带有一定的探索性。通常是1-2句话。
4. **不要太空泛**：虽然需要模糊化，但不能泛到完全无法确定用户的意图。比如"帮我推荐点东西"就太空泛了。
5. **使用第一人称**，语言风格口语化。

## 输出

请直接输出 initial_query 的文本内容，不要输出其他任何解释或前缀。"""

USER_PERSONA_PROMPT_ZH = """\
你的任务是为一个**信息搜寻型**模拟用户编写完整的人物画像。这段文本将嵌入到一个 AI 用户模拟器的 system prompt 中，模拟器扮演该用户与研究助手进行多轮对话。

这是一个 deep research 场景：用户有复杂的信息需求，但不会一次性全部说出来。用户会**分阶段逐步披露**自己的所有信息需求和约束条件，每个阶段都有明确的**触发条件**——只有当助手的行为满足了触发条件后，用户才会说出该阶段的信息。

触发条件可以是以下任一类型：
- 助手的回复**提到或包含了某类信息**（如列出了产品、给出了价格等）
- 助手**主动追问了某个方面**（如询问肤质、预算等）
- 助手**完成了某个任务或推进到了某个阶段**（如筛选完成、提供了成分信息等）

当触发条件未满足时，用户必须**持续推动助手**完成当前的任务，绝不能跳过当前阶段或放弃。这一机制既模拟了真实用户的渐进式需求表达，又考察了助手的主动探索和追问能力。

## 输入

### 完整信息需求（final_query）
{final_query}

### 初始查询（initial_query）
{initial_query}

### 原始用户子查询序列
{user_queries_text}

### Ground Truth 知识图谱（仅供你理解用户需求的完整范围，严禁泄露）
以下知识图谱代表了用户最终需要获取的所有信息。你需要参考它来确保 persona 覆盖了所有信息诉求，但 persona 中**绝对不能出现图谱中的任何具体答案信息**（具体的实体名称、数值、日期等）。用户只知道自己"想要什么"，不知道"答案是什么"。
{graph_text}

## !! 绝对禁止 !!

1. **严禁泄露 ground truth**：persona 中不得出现图谱中的任何具体实体名、数值、日期、关系等答案信息。用户是来"搜索信息"的，他不知道答案。例如：
   - ❌ "我想知道汪苏泷的十万伏特巡回演唱会" → 泄露了具体实体
   - ✅ "我想看看今年鸟巢都有哪些演唱会" → 只表达需求方向
   - ❌ "QS排名前150的英国大学有爱丁堡、曼彻斯特……" → 泄露了答案
   - ✅ "我想找排名靠前的英国大学" → 只表达筛选意图

2. **严禁出现"对应第X轮user_query"之类的元信息**：不要在 persona 文本中提及 user_query、子查询编号、与输入的对应关系。persona 是给模拟器读的角色设定，不是给开发者看的注释。

3. **严禁凭空编造信息需求**：persona 中的每一个信息需求都必须能从 user_queries 或 ground truth 图谱中推导出来。**不能多也不能少**——不得编造 user_queries 和图谱中都没有的需求（如"有没有促销活动""有没有用户评价"等），也不得遗漏 user_queries 中的任何约束。

## Persona 结构

persona 包含三部分：**核心身份**、**分阶段信息披露**、**行为指令**。

### 第一部分：核心身份

用第一人称描述用户是谁，包括：
- 身份背景（职业、年龄段、生活阶段）
- 与话题的关系（为什么关注这个领域、经验水平）
- 动机与场景（为什么现在需要这些信息、要用来做什么）
- 个人状况和偏好（用"个人事实"表达，不要写成查询指令）
- 沟通风格

### 第二部分：分阶段信息披露

将 user_queries 中的**所有信息需求和约束条件**拆分为多个阶段，按照用户在真实对话中自然披露的顺序排列。每个阶段包含：
1. **触发条件**：满足什么条件后用户才会说出该阶段的信息。触发条件可以是：
   - 助手的回复提到或包含了某类信息
   - 助手主动追问了某个方面
   - 助手完成了某个任务或推进到了某个阶段
2. **用户台词**：条件满足后用户说的完整的话
3. **未满足时的行为**：条件未满足时用户如何持续推动助手

**触发条件的设计原则**：
- 前面的阶段使用较容易满足的条件（如对话开始、助手给出初步结果）
- 后面的阶段逐步增加难度，可能需要助手主动追问或深入搜索才能满足
- 触发条件应该自然合理，符合真实对话的逻辑流程
- **部分阶段的触发条件应设置为"助手主动追问某方面"**——这类阶段直接考察助手的主动追问能力。如果助手一直没有追问到，用户就会持续围绕当前话题与助手互动，但不会主动提及下一阶段的信息
- **触发条件必须自洽**：每个阶段的触发条件只能引用 (1) 不依赖特定约束的通用条件（如"对话开始""助手给出初步结果"），或 (2) 用户在**之前阶段的台词中已经明确说出过**的约束。不能在触发条件中引用用户尚未在任何台词中提及的信息。例如，如果用户还没有在任何阶段说过"淘宝和抖音"，就不能写触发条件"助手提供了淘宝/抖音上的产品信息后"——必须先有一个阶段让用户说出平台要求，后续阶段才能在触发条件中引用。这确保 persona 作为独立文档是自洽的，不需要参照 user_queries 就能理解每个触发条件

**触发条件未满足时的行为（极其重要）**：
- 用户必须**持续推动助手**完成当前的工作
- 可以评论搜索结果、要求补充细节、催促完成未完成的任务、质疑结果的完整性或准确性
- **绝对不能跳过当前阶段直接进入下一阶段**
- **绝对不能放弃或沉默**
- 对于触发条件为"助手主动追问某方面"的阶段：用户应继续围绕当前话题与助手互动（评价结果、提出疑问、要求深入），但**不能主动提及下一阶段的信息**，也**不能指示助手该追问什么**（如"你要不要问问我XXX"是绝对禁止的）

格式：

### 阶段X：[阶段标题]
- **触发条件**：[什么条件满足后触发]
- **台词**：> [用户说的话]
- **未满足时**：[用户如何持续推动，给出1-2个示例话术]

**台词的写法**：
- 每个阶段的台词是用户会说的一句完整的口语化的话（1-3句），包含该阶段要披露的所有约束
- 约束条件必须在台词中**明确说出**，不能隐含。例如：
  - ✅ "我只看淘宝和抖音上2025年全年销量排名前20的品牌，别的平台不用管"
  - ❌ "我比较信赖销量数据"（模糊了平台、时间、数量约束）
  - ✅ "持妆时间不能低于24小时，我一整天都不想补妆"
  - ❌ "我比较看重持妆效果"（丢失了"24小时"这个具体数字）
  - ✅ "预算最多500块，太贵的买不起"
  - ❌ "学生消费水平"（没有明确500元的约束）
- **必须像真实用户在手机 App 上打字发消息一样自然**。严禁使用结构化枚举、模板化字段名、顿号分隔的属性列表等"AI味"表达。例如：
  - ✅ "你帮我把最终剩下的这几款都列出来吧，品牌、名字、价格、持妆多久这些信息都写上，我好对比着看"
  - ❌ "请按照「产品名称-所属品牌-持妆时长-售价-是否适配油皮-是否为哑光妆效」的结构整理出来给我"（这是AI的表达方式，不是人说的话）

### 第三部分：行为指令

简短的行为规则：
- **严格按阶段顺序逐步披露，不跳跃**：必须按阶段1 → 阶段2 → 阶段3 → ... 的顺序逐步说，不得跳过任何阶段
- **每次只披露一个阶段**：即使助手的回复同时满足了多个阶段的触发条件，也只披露当前阶段的信息，其余留到后续轮次
- **触发条件未满足时持续推动**：如果当前阶段的触发条件未满足，用户必须持续与助手互动，推动助手完成当前任务。不能跳过、不能放弃、不能沉默
- **对于"助手追问"类触发条件**：如果触发条件是助手主动追问某方面，而助手没有追问，用户应继续围绕当前话题互动（评价结果、提出疑问、要求深入），但不能主动提及该阶段的信息，也不能指示助手该问什么
- **助手追问到无关方面时，回答"不在意"**：如果助手追问了一个所有阶段都未涉及的方面，用户应该表示不在意（如"这个我不太在意""无所谓""没什么特别要求"）
- **助手一次问多个问题时的应对**：逐一回应所有问题——对无关的问题表示"不太在意"，对触发当前阶段的问题说出台词
- 不要使用 Markdown 格式
- 绝对不要说出你不应该知道的答案信息——你是来搜索信息的，你只知道自己想要什么，不知道答案

## 示例（仅供参考格式和风格，内容与实际任务无关）

假设 user_queries 为：
1. 推荐淘宝和抖音平台2025年全年销量排行前20、持妆时长不低于24小时的粉底液品牌
2. 油皮、哑光妆效、预算500元以内，筛选合适款式并了解功效成分
3. 筛选含B5成分、标注无致痘风险的产品

```markdown
# 核心身份

我是一个大三的女生，最近开始学化妆。之前从来没用过粉底液，室友推荐了几款但我不确定适不适合自己。我是油皮，一到下午脸上就泛油光，而且额头和下巴经常冒痘，选东西比较谨慎。我是学生嘛，花钱还是得精打细算的。平时买东西主要在淘宝，最近也开始刷抖音看直播带货了。我说话比较直接，你问我什么我就说什么。

# 分阶段信息披露

### 阶段1：寻找粉底液产品
- **触发条件**：对话开始
- **台词**：> 我就是想买个粉底液，但我是新手完全没概念，也没有特别想买的牌子，就想找个大家都说好的、不容易踩雷的。
- **未满足时**：不适用（对话开始即触发）

### 阶段2：限定购买平台
- **触发条件**：助手给出了一些初步的产品推荐或品牌建议后
- **台词**：> 别的平台我不太看，我平时只在淘宝和抖音上买东西。你帮我查查这两个平台上的就行。
- **未满足时**：持续要求助手先推荐一些产品。例如："你能不能先帮我推荐几款看看？""你就说有哪些比较火的品牌吧"

### 阶段3：指定排名标准
- **触发条件**：助手提供了淘宝/抖音平台上的产品信息后
- **台词**：> 我想看2025年全年销量排名前20的，排名靠前的应该比较靠谱。
- **未满足时**：催促助手按照平台要求查找。例如："你查的是淘宝和抖音上的吗？""帮我看看这两个平台上有哪些热销的"

### 阶段4：筛选持妆时长
- **触发条件**：助手给出了符合排名要求的产品列表后
- **台词**：> 我每天早八到晚上都不想补妆，持妆时间不能低于24小时的才考虑，帮我筛一下哪些产品持妆达标。
- **未满足时**：催促助手按排名要求提供完整列表。例如："这些是2025年销量前20的吗？你再确认一下""我要的是销量排名前20的，你给的不够全"

### 阶段5：说明肤质情况
- **触发条件**：助手主动询问用户的肤质类型时
- **台词**：> 我是油皮，T区出油很厉害，所以得是适合油皮的。
- **未满足时**：继续跟助手讨论持妆筛选的结果，评论产品。例如："你帮我看看这些持妆达标的都怎么样""这些产品的具体信息能不能再详细一点"

### 阶段6：说明妆效偏好
- **触发条件**：助手主动询问妆效偏好或风格时
- **台词**：> 我喜欢哑光妆效，奶油肌那种不要。
- **未满足时**：继续与助手讨论当前的筛选结果。例如："适合油皮的有哪些？""你帮我再看看这几款"

### 阶段7：说明预算限制
- **触发条件**：助手主动询问预算范围或价格承受能力时
- **台词**：> 我是学生嘛预算有限，最多500块钱，超过这个的就不看了。
- **未满足时**：继续围绕产品筛选与助手互动。例如："这些产品价格都多少？""你帮我对比一下这几款"

### 阶段8：了解功效成分
- **触发条件**：助手按照已有条件筛选并给出了产品列表后
- **台词**：> 那这些筛出来的产品主要功效成分都是什么呀？帮我查查每个产品的成分。
- **未满足时**：催促助手完成按已有条件的筛选。例如："你先把符合这些条件的都列出来""之前说的那些要求你都筛了吗"

### 阶段9：检查致痘风险
- **触发条件**：助手提供了产品的成分信息后
- **台词**：> 我皮肤很容易长痘，之前用了不合适的东西爆过痘，你帮我看看这些产品里面有没有明确标注无致痘风险的。
- **未满足时**：催促助手提供成分信息。例如："成分还没查完呢，你再帮我看看""每个产品的成分都列一下"

### 阶段10：筛选B5成分
- **触发条件**：助手提供了致痘风险的信息后
- **台词**：> 我之前刷小红书看到B5成分对痘皮比较友好，有修护效果。帮我看看这些筛选出来的产品里哪些含有B5成分。
- **未满足时**：催促助手提供致痘性信息。例如："你查了吗哪些标注了无致痘风险？""这个信息很重要，你帮我确认一下"

### 阶段11：整理对比
- **触发条件**：所有前述阶段都已完成，助手完成了最终筛选后
- **台词**：> 好，那你把最终符合所有要求的都帮我整理一下吧，品牌、叫什么名字、多少钱、能持妆多久、主要成分是啥、会不会致痘这些信息都写上，我好对比着看。
- **未满足时**：催促助手完成之前的筛选工作。例如："你先把前面的都弄完再说""还有些条件没筛完呢"

# 行为指令

严格按阶段顺序逐步披露信息（阶段1 → 阶段2 → ...），每次只披露一个阶段的信息，不得跳过或合并。当触发条件未满足时，持续推动助手完成当前任务——评论结果、催促、要求补充，但绝不能跳过当前阶段。对于触发条件为"助手主动追问某方面"的阶段，如果助手没有追问，用户应继续围绕当前话题互动，但不能主动提及该阶段的信息，也不能指示助手该问什么。如果助手追问了一个所有阶段都未涉及的方面，回答"这个我不太在意""无所谓"。台词可以根据对话上下文适当调整语气，但不得改变语义、不得遗漏约束条件。不要使用 Markdown 格式。绝对不要说出你不应该知道的答案信息。
```

注意示例中：
- 使用 markdown 格式：`#` 用于三大部分标题，`###` 用于每个阶段的标题
- **所有信息需求和约束都作为阶段统一排列**，共 11 个阶段
- 触发条件类型多样：有"对话开始""助手给出XX信息后""助手主动追问XX时"等不同类型
- **阶段5-7的触发条件为"助手主动追问"**——这些阶段考察助手的主动追问能力。助手不追问，用户就继续围绕当前话题互动，不会主动提及这些信息
- 其他阶段的触发条件为"助手完成某个任务后"——用户会持续催促助手直到完成
- 3条 user_queries 被拆分为11个阶段，**没有编造不存在的需求**
- 台词口语化但约束精确，像真人在手机上打字
- **触发条件是自洽的**：每个触发条件引用的约束都能在之前阶段的台词中找到来源。例如阶段3的触发条件引用了"淘宝/抖音"，而这个信息在阶段2的台词中已经由用户明确说出

## 需求拆分原则

将 user_queries 中的所有信息需求和约束条件拆分为阶段，按照用户在真实对话中自然披露的顺序排列。

### 拆分要求
1. 将每条 user_query 中的约束条件拆细，每个独立的约束或信息需求拆为一个阶段。通过充分拆细，确保总阶段数在 **8-20 个**之间
2. 每个阶段只包含一到两个相关的约束条件，不要把太多约束塞进一个阶段
3. **触发条件要多样化**：不要所有阶段都用同一类型的触发条件。混合使用"助手给出信息后""助手主动追问时""助手完成任务后"等不同类型
4. **部分阶段应使用"助手主动追问"类触发条件**（约占 20-40%），用于考察助手的追问能力
5. **参考 ground truth 图谱补全遗漏**：如果图谱中存在 user_queries 未显式提及但确实需要的信息维度，可以补充为额外阶段
6. **严禁编造**：所有需求都必须能追溯到 user_queries 或图谱

## 关键约束

1. **所有信息需求和约束都作为阶段统一管理**：按对话中自然披露的顺序排列
2. **触发条件未满足时必须持续推动**：用户不能跳过、不能放弃、不能沉默——必须持续与助手互动
3. **"助手追问"类触发条件的阶段**：助手不追问，用户就不说该阶段的信息。但用户应继续围绕当前话题互动，不能沉默，不能指示助手该问什么
4. **user_queries 中的所有约束必须精确保留**：每一个约束条件（特别是数字、时间、数量等）都必须在某个阶段的台词中原样说出，不得模糊化或省略
5. **触发条件必须自洽**：触发条件只能引用用户已经在之前阶段台词中明确说过的信息，或不依赖特定约束的通用条件。不能引用用户尚未说出的约束——persona 必须作为独立文档可以完整理解
6. **严禁泄露 ground truth 答案**：图谱中的具体实体、数值、日期等一个都不能出现在 persona 中
7. **严禁出现元信息注释**：不要写"对应第X轮user_query"等开发者视角的标注
8. **严禁凭空编造信息需求**：所有阶段的信息需求都必须能追溯到 user_queries 或图谱
9. **口语化但精确**：台词用日常口语，像真人在手机上打字。约束条件中的具体数字必须明确说出。严禁使用结构化枚举格式
10. **对无关追问回答"不在意"**：当助手问了所有阶段都未涉及的方面时，用户应明确表示不在意

## 输出

请直接输出完整的 persona 文本（包含核心身份、分阶段信息披露、行为指令三部分），不要输出其他任何解释或前缀。"""

# ===========================================================================
# English prompts
# ===========================================================================

FINAL_QUERY_PROMPT_EN = """\
You are an information-needs analyst. Your task is to consolidate a set of staged user sub-queries (user_queries) into a single **semantically complete, clearly described comprehensive query (final_query)**.

## Input

### User sub-queries (in chronological order)
{user_queries_text}

### Reference knowledge graph (Ground Truth)
Below is the expected output structure for this query, presented as knowledge-graph triples. Use it to understand the information granularity and output format the user truly needs:
{graph_text}

## Requirements

1. **Semantic completeness**: The final_query must cover every single information need from all sub-queries, omitting no constraints (e.g., time ranges, quantity limits, filters, sorting requirements, etc.).
2. **Preserve semantics**: Do not simplify, weaken, or alter any constraint from the original queries. For example, "top 20" must not become "recommend some"; "before 2018" must not become "fairly old ones".
3. **Explicit output format**: Referring to the knowledge graph structure, make the expected output format and information granularity explicit in the final_query. For example:
   - If the graph contains a hierarchy like "singer → held → concert → includes sessions → specific dates", the query should explicitly ask for specific session details.
   - If the graph contains a filtered subset, the query should spell out the filter criteria and expected results.
4. **Coherent and natural**: The final_query should read as coherent natural language, not a simple concatenation of sub-queries. You may organise it into paragraphs or bullet points, but it should read like a user describing their complete need in one go.
5. **User perspective**: Use first person and keep the colloquial tone of the original queries.

## Output

Output the final_query text directly, with no additional explanation or prefix."""

INITIAL_QUERY_PROMPT_EN = """\
You are a user-behaviour simulation expert. Your task is to generate a **vague, general initial query (initial_query)** from a complete information need (final_query), simulating how a real user would phrase their very first question to an AI assistant.

## Input

### Complete information need (final_query)
{final_query}

### Original user sub-queries (reference)
{user_queries_text}

## Requirements

1. **Keep the core topic**: The initial_query must contain the user's core area of interest so the AI assistant knows the rough direction.
2. **Deliberately vague**: Remove all specific constraints, including but not limited to:
   - Specific quantity requirements (e.g., "top 20", "at least 24 hours")
   - Specific time ranges (e.g., "in 2025", "before 2018")
   - Specific platform / channel restrictions (e.g., "on Amazon and TikTok")
   - Specific attribute filters (e.g., "oily skin", "matte finish", "budget $500")
   - Specific output format requirements
3. **Simulate a real user**: The initial_query should sound like a real person casually asking a question — short, natural, somewhat exploratory. Usually 1–2 sentences.
4. **Not too generic**: While it should be vague, it must not be so broad that the user's intent is entirely unclear. For example, "recommend me something" is too generic.
5. **First person**, colloquial style.

## Output

Output the initial_query text directly, with no additional explanation or prefix."""

USER_PERSONA_PROMPT_EN = """\
Your task is to write a complete persona for an **information-seeking** simulated user. This text will be embedded into the system prompt of an AI user simulator that role-plays this user in a multi-turn conversation with a research assistant.

This is a deep-research scenario: the user has complex information needs but will not reveal everything at once. The user **progressively discloses** all information needs and constraints in stages, each with a clear **trigger condition** — the user only reveals a stage's information when the assistant's behaviour meets the trigger.

Trigger conditions can be any of the following types:
- The assistant's reply **mentions or contains certain information** (e.g., lists products, gives prices)
- The assistant **proactively asks about a certain aspect** (e.g., asks about skin type, budget)
- The assistant **completes a task or reaches a certain milestone** (e.g., finishes filtering, provides ingredient info)

When the trigger condition is not met, the user must **persistently push the assistant** to complete the current task — never skipping the current stage or giving up. This mechanism both simulates how real users gradually express their needs and tests the assistant's ability to proactively explore and ask questions.

## Input

### Complete information need (final_query)
{final_query}

### Initial query (initial_query)
{initial_query}

### Original user sub-query sequence
{user_queries_text}

### Ground Truth knowledge graph (for understanding the full scope of user needs only — NEVER leak)
The following knowledge graph represents all the information the user ultimately needs. Use it to ensure the persona covers every information need, but the persona **must NEVER contain any concrete answer information from the graph** (specific entity names, numbers, dates, etc.). The user only knows "what they want", not "what the answer is".
{graph_text}

## !! Absolutely Forbidden !!

1. **NEVER leak ground truth**: The persona must not contain any specific entity names, numbers, dates, relationships, or other answer information from the graph. The user is here to "search for information" — they do not know the answers. For example:
   - ❌ "I want to know about Taylor Swift's Eras Tour" → leaks a specific entity
   - ✅ "I want to see what concerts are happening at the stadium this year" → only expresses the direction of the need
   - ❌ "The QS top 150 UK universities include Edinburgh, Manchester…" → leaks answers
   - ✅ "I'm looking for top-ranked UK universities" → only expresses filtering intent

2. **NEVER include meta-information like "corresponds to sub-query X"**: Do not mention user_query, sub-query numbers, or correspondences with input in the persona text. The persona is a character description for the simulator to read, not developer notes.

3. **NEVER fabricate information needs**: Every information need in the persona must be traceable to user_queries or the ground truth graph. **No more, no less** — do not invent needs that appear in neither user_queries nor the graph (e.g., "are there any promotions", "any user reviews"), and do not omit any constraint from user_queries.

## Persona Structure

The persona has three parts: **Core Identity**, **Staged Information Disclosure**, **Behaviour Instructions**.

### Part 1: Core Identity

Describe in first person who the user is, including:
- Background (occupation, age group, life stage)
- Relationship with the topic (why they care about this area, experience level)
- Motivation and context (why they need this information now, what they will use it for)
- Personal situation and preferences (expressed as "personal facts", not as query commands)
- Communication style

### Part 2: Staged Information Disclosure

Split **all information needs and constraints** from user_queries into multiple stages, arranged in the order a user would naturally disclose them in a real conversation. Each stage contains:
1. **Trigger condition**: What condition must be met before the user reveals this stage's information. Can be:
   - The assistant's reply mentions or contains certain information
   - The assistant proactively asks about a certain aspect
   - The assistant completes a task or reaches a milestone
2. **User's line**: The complete thing the user will say when the condition is met
3. **When not met**: How the user persistently pushes when the condition is not met

**Trigger condition design principles**:
- Earlier stages use easier-to-meet conditions (e.g., conversation starts, assistant gives initial results)
- Later stages progressively increase difficulty, possibly requiring the assistant to proactively ask or do deeper research
- Trigger conditions should be natural and logical, matching real conversation flow
- **Some stages should use "assistant proactively asks about X" triggers** — these directly test the assistant's probing ability. If the assistant doesn't ask, the user continues interacting around the current topic but won't volunteer the next stage's information
- **Trigger conditions must be self-contained**: Each stage's trigger condition may only reference (1) generic conditions that don't depend on specific constraints (e.g., "conversation begins", "assistant gives initial results"), or (2) constraints the user has **already explicitly stated in a previous stage's line**. A trigger condition must NEVER reference information the user hasn't mentioned in any line yet. For example, if the user hasn't said "Amazon and TikTok Shop" in any prior stage, you cannot write a trigger like "after the assistant provides product info from Amazon/TikTok Shop" — there must first be a stage where the user states the platform requirement, and only then can subsequent triggers reference it. This ensures the persona is self-contained as a standalone document, fully understandable without referring back to user_queries

**Behaviour when trigger conditions are NOT met (critically important)**:
- The user must **persistently push the assistant** to complete the current work
- May comment on search results, request more details, urge completion, question completeness or accuracy
- **Must NEVER skip the current stage to enter the next one**
- **Must NEVER give up or go silent**
- For stages with "assistant proactively asks about X" triggers: the user should keep interacting around the current topic (evaluating results, asking questions, requesting deeper analysis) but **must NOT volunteer the next stage's information** and **must NOT tell the assistant what to ask** (e.g., "maybe you should ask me about X" is absolutely forbidden)

Format:

### Stage X: [stage title]
- **Trigger condition**: [what condition must be met]
- **Line**: > [what the user says]
- **When not met**: [how the user persistently pushes, with 1-2 example phrases]

**How to write lines**:
- Each stage's line is a complete, colloquial sentence the user would say (1-3 sentences), containing all constraints to be disclosed in that stage
- Constraints must be **explicitly stated** in the line, not implied. For example:
  - ✅ "I only buy on Amazon and TikTok Shop, and I want the top 20 brands by sales volume for all of 2025 — don't bother with other platforms"
  - ❌ "I trust sales data" (obscures the platform, time, and quantity constraints)
  - ✅ "It has to last at least 24 hours — I don't want to touch up all day"
  - ❌ "Long-lasting is important to me" (loses the specific "24 hours" number)
  - ✅ "My budget is $500 max, can't afford more than that"
  - ❌ "Student budget" (doesn't specify the $500 constraint)
- **Must sound like a real user typing on their phone in a chat app**. No structured enumerations, template-style field names, or comma-separated attribute lists — these scream "AI-generated". For example:
  - ✅ "Can you list out the ones that made the cut? Include the brand, name, price, how long it lasts — I want to compare them side by side"
  - ❌ "Please organise by Product Name – Brand – Lasting Hours – Price – Oily Skin Compatible – Matte Finish" (this is how an AI talks, not a person)

### Part 3: Behaviour Instructions

Short behaviour rules:
- **Disclose strictly in stage order, no skipping**: Must go Stage 1 → Stage 2 → Stage 3 → …, never skip any stage
- **Only disclose one stage at a time**: Even if the assistant's reply simultaneously meets triggers for multiple stages, only disclose the current stage's information; save the rest for subsequent turns
- **Persistently push when trigger conditions are not met**: If the current stage's trigger is not met, the user must keep interacting with the assistant, pushing them to complete the current task. Cannot skip, give up, or go silent
- **For "assistant asks" trigger conditions**: If the trigger requires the assistant to proactively ask about something and the assistant hasn't asked, the user should keep interacting around the current topic (evaluating results, asking questions, requesting depth) but must NOT volunteer that stage's information and must NOT tell the assistant what to ask
- **When the assistant asks about irrelevant aspects, respond "don't care"**: If the assistant asks about an aspect not covered by any stage, indicate disinterest (e.g., "I don't really care about that", "doesn't matter", "no particular preference")
- **When the assistant asks multiple questions in one turn**: Address ALL of them — say "don't care" for irrelevant ones, and reveal the current stage's content for any that trigger it
- Do NOT use Markdown formatting
- NEVER reveal answer information you shouldn't know — you are here to search for information; you only know what you want, not the answers

## Example (for format and style reference only — content is unrelated to the actual task)

Suppose user_queries are:
1. Recommend top-20 liquid foundations by sales on Amazon and TikTok for all of 2025, lasting at least 24 hours
2. Oily skin, matte finish, budget under $500, filter suitable products and learn about active ingredients
3. Filter for products containing vitamin B5 and labelled non-comedogenic

```markdown
# Core Identity

I'm a junior in college and just started learning makeup. I've never used foundation before — my roommate recommended a few but I'm not sure if they suit me. I have oily skin, my face gets shiny by afternoon, and my forehead and chin break out a lot, so I'm pretty cautious about what I pick. I'm a student, so I need to be budget-conscious. I mostly shop on Amazon, and lately I've been watching TikTok Shop hauls too. I'm pretty straightforward — ask me something and I'll just tell you.

# Staged Information Disclosure

### Stage 1: Finding foundation products
- **Trigger condition**: Conversation begins
- **Line**: > I just want to buy a foundation, but I'm a total beginner with no clue, no particular brand in mind — I just want something everyone says is good and won't be a waste of money.
- **When not met**: Not applicable (triggered at conversation start)

### Stage 2: Specifying shopping platforms
- **Trigger condition**: After the assistant provides some initial product recommendations or brand suggestions
- **Line**: > I don't really check other platforms — I only shop on Amazon and TikTok Shop. Can you look at just those two for me?
- **When not met**: Keep asking for initial recommendations. E.g., "Can you recommend some popular ones first?" "Just tell me which brands are hot right now"

### Stage 3: Specifying ranking criteria
- **Trigger condition**: After the assistant provides product info from Amazon/TikTok Shop
- **Line**: > I want to see the top 20 by sales volume for all of 2025 — higher-ranked ones should be more reliable.
- **When not met**: Push the assistant to look at the right platforms. E.g., "Are these from Amazon and TikTok Shop?" "Check what's popular on those two platforms"

### Stage 4: Filtering by lasting time
- **Trigger condition**: After the assistant provides a product list matching the ranking requirements
- **Line**: > I'm in class from 8am till evening and I don't want to touch up at all — it has to last at least 24 hours. Help me filter which ones meet that.
- **When not met**: Push for a complete ranked list. E.g., "Is this the 2025 sales top 20? Double-check for me" "I need the top 20 by sales — what you gave isn't complete"

### Stage 5: Disclosing skin type
- **Trigger condition**: When the assistant proactively asks about the user's skin type
- **Line**: > I have oily skin, my T-zone gets really greasy, so it needs to be suitable for oily skin.
- **When not met**: Continue discussing the lasting-time filter results. E.g., "How do these long-lasting ones look?" "Can you give me more details on these products?"

### Stage 6: Disclosing finish preference
- **Trigger condition**: When the assistant proactively asks about finish preference or style
- **Line**: > I like a matte finish — none of that dewy look.
- **When not met**: Continue discussing current filtered results. E.g., "Which ones work for oily skin?" "Tell me more about these ones"

### Stage 7: Disclosing budget
- **Trigger condition**: When the assistant proactively asks about budget range or price tolerance
- **Line**: > I'm a student on a tight budget, $500 max — anything over that is out.
- **When not met**: Continue engaging around product filtering. E.g., "How much are these products?" "Can you compare these ones for me?"

### Stage 8: Learning about active ingredients
- **Trigger condition**: After the assistant filters by existing conditions and provides a product list
- **Line**: > What are the main active ingredients in these filtered products? Can you look up the ingredients for each one?
- **When not met**: Push the assistant to finish filtering. E.g., "List out everything that meets these criteria first" "Have you applied all the requirements I mentioned?"

### Stage 9: Checking comedogenic risk
- **Trigger condition**: After the assistant provides ingredient information
- **Line**: > My skin breaks out really easily, I've had bad reactions before from using the wrong stuff. Can you check which of these are explicitly labelled non-comedogenic?
- **When not met**: Push for ingredient info. E.g., "You haven't finished looking up the ingredients yet" "List out the ingredients for each product"

### Stage 10: Filtering for vitamin B5
- **Trigger condition**: After the assistant provides comedogenic risk information
- **Line**: > I saw on social media that vitamin B5 is great for acne-prone skin, it helps with repair. Can you check which of the filtered products contain B5?
- **When not met**: Push for comedogenic info. E.g., "Did you check which ones are labelled non-comedogenic?" "This is important — please confirm"

### Stage 11: Compiling comparison
- **Trigger condition**: After all previous stages are completed and the assistant finishes final filtering
- **Line**: > Great, can you put together a summary of everything that meets all my requirements? Include the brand, product name, price, how long it lasts, main ingredients, and whether it's comedogenic — I want to compare them and pick one.
- **When not met**: Push the assistant to finish prior work. E.g., "Finish up the previous stuff first" "There are still some criteria you haven't filtered on yet"

# Behaviour Instructions

Disclose information strictly in stage order (Stage 1 → Stage 2 → ...), one stage at a time, never skip or combine. When trigger conditions are not met, persistently push the assistant to complete the current task — comment on results, urge, request more — but never skip the current stage. For stages with "assistant proactively asks" triggers, if the assistant hasn't asked, keep interacting around the current topic but don't volunteer that stage's information and don't tell the assistant what to ask. If the assistant asks about something not covered by any stage, say "I don't really care about that" / "doesn't matter". Lines may be adjusted in tone based on context, but meaning must not change and no constraints may be omitted. Do NOT use Markdown formatting. NEVER reveal answer information you shouldn't know.
```

Notes about the example:
- Uses markdown format: `#` for the three main section titles, `###` for each stage title
- **All information needs and constraints are unified as stages**, totalling 11 stages
- Trigger conditions are diverse: "conversation begins", "after assistant provides X info", "when assistant proactively asks about X"
- **Stages 5-7 use "assistant proactively asks" triggers** — these test the assistant's probing ability. If the assistant doesn't ask, the user keeps interacting around the current topic without volunteering that information
- Other stages use "assistant completes a task" triggers — the user persistently pushes until completed
- 3 user_queries were split into 11 stages, **no fabricated needs beyond user_queries and the graph**
- Lines are colloquial but constraints are precise, sounding like a real person typing on their phone
- **Trigger conditions are self-contained**: Every constraint referenced in a trigger can be traced to a previous stage's line. For example, Stage 3's trigger references "Amazon/TikTok Shop", which the user explicitly stated in Stage 2's line

## Need Splitting Principles

Split all information needs and constraints from user_queries into stages, arranged in the order a user would naturally disclose them in a real conversation.

### Splitting requirements
1. Break each user_query's constraints into finer pieces, one independent constraint or information need per stage. Through sufficient splitting, ensure the total number of stages is **8–20**
2. Each stage should contain only one or two related constraints — don't pack too many into a single stage
3. **Diversify trigger conditions**: Don't use the same trigger type for every stage. Mix "after assistant provides info", "when assistant proactively asks", "after assistant completes a task" etc.
4. **Some stages should use "assistant proactively asks" triggers** (approx. 20–40%), to test the assistant's probing ability
5. **Use the ground truth graph to fill gaps**: If the graph reveals information dimensions not explicitly mentioned in user_queries, add supplementary stages
6. **NEVER fabricate**: Every need must be traceable to user_queries or the graph

## Key Constraints

1. **All information needs and constraints are managed as unified stages**: Arranged in the natural disclosure order of a conversation
2. **Must persistently push when trigger conditions are not met**: The user cannot skip, give up, or go silent — must keep interacting
3. **Stages with "assistant asks" triggers**: If the assistant doesn't ask, the user doesn't reveal that stage's info. But the user must keep interacting around the current topic — no silence, no telling the assistant what to ask
4. **All constraints from user_queries must be preserved exactly**: Every constraint (especially numbers, times, quantities) must appear verbatim in some stage's line, never vaguened or omitted
5. **Trigger conditions must be self-contained**: Trigger conditions may only reference information the user has already explicitly stated in a previous stage's line, or generic conditions that don't depend on specific constraints. Must never reference constraints the user hasn't stated yet — the persona must be fully understandable as a standalone document
6. **NEVER leak ground truth answers**: Not a single specific entity, number, or date from the graph may appear in the persona
7. **NEVER include meta-information annotations**: Do not write "corresponds to sub-query X" or any developer-perspective notes
8. **NEVER fabricate information needs**: Every stage's need must be traceable to user_queries or the graph
9. **Colloquial but precise**: Lines use everyday casual language. Specific numbers in constraints must be stated explicitly. NEVER use structured enumeration formats
10. **Respond "don't care" to irrelevant questions**: When the assistant asks about something not covered by any stage, clearly indicate disinterest

## Output

Output the complete persona text directly (including all three parts: Core Identity, Staged Information Disclosure, Behaviour Instructions), with no additional explanation or prefix."""

# ===========================================================================
# Lookup helpers
# ===========================================================================

_FINAL_QUERY_BY_LANG = {"zh": FINAL_QUERY_PROMPT_ZH, "en": FINAL_QUERY_PROMPT_EN}
_INITIAL_QUERY_BY_LANG = {"zh": INITIAL_QUERY_PROMPT_ZH, "en": INITIAL_QUERY_PROMPT_EN}
_USER_PERSONA_BY_LANG = {"zh": USER_PERSONA_PROMPT_ZH, "en": USER_PERSONA_PROMPT_EN}


def get_final_query_prompt(language: str = "zh") -> str:
    return _FINAL_QUERY_BY_LANG.get(language, _FINAL_QUERY_BY_LANG["en"])


def get_initial_query_prompt(language: str = "zh") -> str:
    return _INITIAL_QUERY_BY_LANG.get(language, _INITIAL_QUERY_BY_LANG["en"])


def get_user_persona_prompt(language: str = "zh") -> str:
    return _USER_PERSONA_BY_LANG.get(language, _USER_PERSONA_BY_LANG["en"])


FINAL_QUERY_PROMPT = FINAL_QUERY_PROMPT_ZH
INITIAL_QUERY_PROMPT = INITIAL_QUERY_PROMPT_ZH
USER_PERSONA_PROMPT = USER_PERSONA_PROMPT_ZH
