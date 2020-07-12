<template>
    <div class="document">
        <!-- eslint-disable -->
        <h3 align="center">
            Summarization Highlight
        </h3>
        <p>This task requires <strong>three</strong> steps:</p>
        <p><strong>Step 1: Highlight the Main Points a Summary Mentioned in a Document</strong></p>
        <p><strong>A summary is</strong> a brief statement or account of the main points of a document. In this step, you will be given two components <strong>a summary</strong> and <strong>a document</strong>. You should:<br/></p>
            <blockquote>
                1) Read the summary and the document <br/>
                2) Highlight all these main points the summary mentioned in the document <br/>
                3) Check your highlights by answing the question "Do those highlights cover as much the main points mentioned in the summary as possible?" <br/>
            </blockquote>
        <p>An example of highlights:</p>
            <blockquote>
                Summary:<br/>
                <em>A council plans to employ its own staff to help young people with mental health problems.</em><br/><br/>
                Document:<br/>
                ...<em><a class="highlight">West Berkshire Council</a> is setting up an <a class="highlight">emotional health</a> academy to train psychology graduates and health professionals. The local authority said , once trained , <a class="highlight">its staff will work with children , families and schools</a>. It wants to greatly reduce the wait <a class="highlight">mental health</a> patients face from 12 months to less than a week... </em><br/>
            </blockquote>

        <p>The maximum combined length of all highlighted phrases is <strong>30 words</strong> . If you feel that 30 words is not enough to highlight all important information, then highlight only the most important parts that fit within this limit. The task should be possible to complete in less than 10 minutes.</p>
        <hr/>
        <p><strong>HINT</strong>: please prioritize sentences or long phrases that can describe one or more main points to highlight first. If you can find neither sentences nor long phrases to describe any of the main points the summary described, you can highlight short phrases and words. </p>
        <hr/>
        <p><strong>Step 2: Comprehension Question</strong></p>
        <p><strong>Please read the whole document before you do the highlighting</strong>. After you have submitted the highlight, there will be one simple True/False question where we will verify whether you have read the document or not. Failure to answer the question correctly will invalidate the evaluation that the participant has done.</p>
        
        
        <hr/><p><strong>Step 3: Can you extract all the main points present in the summary from the document?</strong></p>
        <p>Following the Comprehension Question, there is a track bar. Set a value from 1 to 5 by moving the indicator to show how many main points in the summary can be extracted from the document: </p>
        <blockquote>
                <strong>5</strong>: All the main points in the summary are explicitly described in the document. <br/>
                <strong>4</strong>: Most of the main points in the summary are explicitly described in the document. <br/>
                <strong>3</strong>: Most main points are not explicitly mentioned in the document. However, most of them can be deduced from the document. <br/>
                <strong>2</strong>: Most main points are neither explicitly mentioned nor can be deduced from the document. However both texts fit under the same news story.  <br/>
                <strong>1</strong>: The texts are not related at all. <br/>
        </blockquote>
        
        <hr/><p>There is a <strong>code</strong> that you have to copy and paste to Amazon platform at the end of the task. Only submissions that correctly answer the True/False question will be deemed as a valid HIT.</p>
        <hr/>
        <p>Thank you for reading this</p>

        <p>
            Do you consent for all the information that you have submitted to be used
            anonymously for research purposes only?
        </p>
        <!-- eslint-enable -->
    </div>
</template>

<script>
export default {
  name: 'LandingHighlightMTurk',
};
</script>
<style scoped lang="scss">
h3 {
    margin-top: 2rem;
    margin-bottom: 3rem;
}
.document {
  font-family: 'Lora', serif;
  font-size: 1.1rem;
  line-height: 1.5rem;
  padding-left: 2rem;
  padding-right: 2rem;
  padding-bottom: 2rem;
}
.highlight{
  background-color: greenyellow;
}
</style>
